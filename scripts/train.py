"""Обучение нейросети синтеза ДОЭ (без учителя, функция потерь через модель дифракции).

Пример (GPU):
    python scripts/make_dataset.py --split val --count 64 --seed 1
    python scripts/train.py --steps 20000 --batch 16 --out runs/main

Быстрая проверка на CPU:
    python scripts/train.py --steps 200 --batch 4 --base 16 --out runs/smoke
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from doe.config import DISTANCES
from doe.losses import doe_loss
from doe.model import DOENet
from doe.optics_torch import FresnelPropagator
from doe.torch_data import FolderSets, RandomSets
from doe import registry


def build_model(cfg, device):
    prop = FresnelPropagator(DISTANCES)
    return DOENet(prop, base=cfg["base"], depth=cfg["depth"]).to(device)


@torch.no_grad()
def validate(model, loader, device, w):
    model.eval()
    agg, n = {}, 0
    for t in loader:
        t = t.to(device)
        amp = model.prop(model(t)).abs()
        _, st = doe_loss(amp, t, **w)
        for k, v in st.items():
            agg[k] = agg.get(k, 0.0) + v * len(t)
        n += len(t)
    model.train()
    return {k: v / n for k, v in agg.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "runs" / "main"))
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--w-pcc", type=float, default=1.0)
    ap.add_argument("--w-mse", type=float, default=1.0)
    ap.add_argument("--w-ssim", type=float, default=2.0)
    ap.add_argument("--extra-dir", default=None, help="свои изображения для подмешивания в обучение")
    ap.add_argument("--extra-prob", type=float, default=0.5, help="доля изображений из --extra-dir")
    ap.add_argument("--val-dir", default=str(ROOT / "data" / "val"))
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--ckpt-every", type=int, default=5,
                    help="каждые N валидаций: чекпойнт + метрики на тестовом наборе (0 = выкл.)")
    ap.add_argument("--test-dir", default=str(ROOT / "data" / "test"))
    ap.add_argument("--test-limit", type=int, default=None, help="сколько тестовых наборов использовать")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None, help="продолжить прерванное обучение из last.pt (веса + оптимизатор + шаг)")
    ap.add_argument("--init", default=None, help="дообучение: взять веса из чекпойнта (напр. weights/doe_nn.pt), оптимизатор и расписание LR с нуля")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    # FFT на MPS поддерживается не во всех версиях; при ошибке запускайте с CPU/CUDA
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = {"base": a.base, "depth": a.depth, "distances": list(DISTANCES)}
    w = {"w_pcc": a.w_pcc, "w_mse": a.w_mse, "w_ssim": a.w_ssim}
    (out / "config.json").write_text(json.dumps({**cfg, **w, **vars(a)}, indent=2, ensure_ascii=False), encoding="utf-8")

    model = build_model(cfg, device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
    step0 = 0
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location=device)["model"])
        print(f"веса загружены из {a.init}")
    if a.resume:
        ck = torch.load(a.resume, map_location=device)
        model.load_state_dict(ck["model"])
        if "opt" not in ck:
            raise SystemExit("в чекпойнте нет состояния оптимизатора; для дообучения используйте --init")
        opt.load_state_dict(ck["opt"])
        step0 = ck["step"]
        for _ in range(step0):
            sched.step()
    print(f"device={device}, params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    train_loader = DataLoader(RandomSets(a.extra_dir, extra_prob=a.extra_prob, seed=a.seed), batch_size=a.batch,
                              num_workers=a.workers, pin_memory=device == "cuda",
                              persistent_workers=a.workers > 0)
    val_loader = None
    if Path(a.val_dir).exists():
        val_loader = DataLoader(FolderSets(a.val_dir), batch_size=a.batch)
    else:
        print(f"[!] нет {a.val_dir}: создайте его через scripts/make_dataset.py, валидация отключена")

    # --- контроль деградации: фиксированный тестовый набор и эталон (опубликованная модель) ---
    test_sets = None
    if a.ckpt_every > 0 and Path(a.test_dir).exists():
        test_sets = registry.load_sets(a.test_dir, a.test_limit)
        (out / "checkpoints").mkdir(exist_ok=True)
    ref = registry.get(registry.load())
    ref_pcc = ref["metrics"]["PCC"] if ref else None
    if ref:
        print(f"эталон: {ref['version']} PCC={ref_pcc:.4f} (порог деградации {registry.DEGRADATION_TOL})")
    hist_path = out / "metrics_history.csv"
    new_hist = not hist_path.exists()
    hist = open(hist_path, "a", newline="", encoding="utf-8")
    hw = csv.writer(hist)
    if new_hist:
        hw.writerow(["step", "val_pcc", "val_ssim", "test_pcc", "test_ssim", "best_test_pcc", "ref_pcc", "status", "ckpt"])
    n_val = 0
    best_test = -1e9

    log = open(out / "log.csv", "a", newline="", encoding="utf-8")
    wr = csv.writer(log)
    if step0 == 0:
        wr.writerow(["step", "split", "loss", "pcc", "ssim", "mse", "lr", "sec"])
    best = -1e9
    t0 = time.time()
    it = iter(train_loader)
    for step in range(step0 + 1, a.steps + 1):
        t = next(it).to(device, non_blocking=True)
        amp = model.prop(model(t)).abs()
        loss, st = doe_loss(amp, t, **w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 50 == 0:
            lr = sched.get_last_lr()[0]
            wr.writerow([step, "train", st["loss"], st["pcc"], st.get("ssim", ""), st["mse"], lr, time.time() - t0])
            print(f"{step:6d} loss={st['loss']:.4f} pcc={st['pcc']:.4f} ssim={st.get('ssim', 0):.4f} lr={lr:.2e}")
        if step % a.val_every == 0 or step == a.steps:
            ck = {"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "cfg": cfg}
            torch.save(ck, out / "last.pt")
            if val_loader is not None:
                v = validate(model, val_loader, device, w)
                wr.writerow([step, "val", v["loss"], v["pcc"], v.get("ssim", ""), v["mse"], "", time.time() - t0])
                score = v["pcc"] + v.get("ssim", 0.0)
                print(f"   VAL pcc={v['pcc']:.4f} ssim={v.get('ssim', 0):.4f}")
                if score > best:
                    best = score
                    torch.save(ck, out / "best.pt")
                n_val += 1
                if test_sets is not None and (n_val % a.ckpt_every == 0 or step == a.steps):
                    tm = registry.benchmark_model(model, test_sets, device)
                    cp = out / "checkpoints" / f"ckpt_step{step:06d}.pt"
                    torch.save({"model": model.state_dict(), "cfg": cfg, "step": step, "test": tm}, cp)
                    status = "ok"
                    if tm["PCC"] < best_test - registry.DEGRADATION_TOL:
                        status = "degraded_vs_best"
                    if ref_pcc is not None and tm["PCC"] < ref_pcc - registry.DEGRADATION_TOL:
                        status = "degraded_vs_release"
                    best_test = max(best_test, tm["PCC"])
                    hw.writerow([step, v["pcc"], v.get("ssim", ""), tm["PCC"], tm["SSIM"], best_test,
                                 ref_pcc if ref_pcc is not None else "", status, cp.name])
                    hist.flush()
                    mark = "" if status == "ok" else "   <-- ДЕГРАДАЦИЯ"
                    print(f"   TEST pcc={tm['PCC']:.4f} ssim={tm['SSIM']:.4f} [{status}] -> {cp.name}{mark}")
            log.flush()
    log.close()
    hist.close()
    try:
        from scripts.plot_history import plot
        plot(hist_path)
    except Exception as e:  # график не критичен
        print("график не построен:", e)
    if not (out / "best.pt").exists():
        torch.save(ck, out / "best.pt")
    print("готово:", out / "best.pt")


if __name__ == "__main__":
    main()
