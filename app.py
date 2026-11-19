"""
Application web Flask : CHROMIA AI.

Fonctionnalités :
    - colorisation automatique
    - prédiction U-Net
    - variantes MC-Dropout
    - carte d'incertitude
    - ajustement saturation
    - ajustement température
    - gain automatique de chrominance
    - cache LRU
    - endpoint de santé

Lancement :

    python app.py

Puis :

    http://127.0.0.1:5000

Mode debug :

    COLORVISION_DEBUG=1 python app.py
"""

import io
import json
import os
import time
import uuid

from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from flask import (
    Flask,
    abort,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)

from PIL import (
    Image,
    UnidentifiedImageError,
)

from werkzeug.utils import secure_filename

from common import (
    IMG_EXT,
    auto_gain,
    compose_rgb,
    mean_chroma,
    predict_ab,
    uncertainty_map,
)

from unet import (
    UNet,
    load_model,
)

from i18n import LANGS, pick_lang, translator

from auth import (
    init_app as init_auth,
    login_required,
    public_config,
    current_user,
    db,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_FOLDER = (
    BASE_DIR
    / "static"
    / "uploads"
)

MODEL_PATH = Path(
    os.environ.get(
        "COLORVISION_MODEL",
        BASE_DIR
        / "checkpoints"
        / "best.pt",
    )
)

# Nombre de variantes MC-Dropout.
N_VARIANTS = 4

# Nombre maximum d'images conservées en mémoire.
CACHE_SIZE = 32

# Taille maximale d'un upload.
MAX_UPLOAD_MB = 20

# ---------------------------------------------------------------------------
# Chrominance cible.
#
# 0 = désactive le gain automatique.
# ---------------------------------------------------------------------------

TARGET_CHROMA = float(
    os.environ.get(
        "COLORVISION_TARGET_CHROMA",
        18,
    )
)

# ---------------------------------------------------------------------------
# Création du dossier uploads.
# ---------------------------------------------------------------------------

UPLOAD_FOLDER.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================================
# FLASK
# ============================================================================

app = Flask(__name__)

app.config[
    "UPLOAD_FOLDER"
] = str(UPLOAD_FOLDER)

app.config[
    "MAX_CONTENT_LENGTH"
] = MAX_UPLOAD_MB * 1024 * 1024

# Comptes utilisateurs : e-mail, Google, Facebook (voir auth.py).
init_auth(app)


# ============================================================================
# CHARGEMENT DU MODÈLE
# ============================================================================

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

infer_size = 256

model = None


if MODEL_PATH.exists():

    try:

        model, ckpt = load_model(
            str(MODEL_PATH),
            device,
        )

        # Récupération de la taille utilisée
        # pendant l'entraînement.
        infer_size = (
            ckpt
            .get("args", {})
            .get("size", 256)
        )

        print(
            f"Modèle chargé depuis "
            f"{MODEL_PATH}"
        )

        print(
            f"Entrée réseau : "
            f"{infer_size}x{infer_size}"
        )

        print(
            f"Device : {device}"
        )

    except Exception as e:

        print(
            "Erreur lors du chargement "
            f"du checkpoint : {e}"
        )

        print(
            "U-Net non entraîné utilisé."
        )

        model = UNet().to(device)

else:

    model = UNet().to(device)

    print(
        "ATTENTION : aucun modèle entraîné "
        "trouvé."
    )

    print(
        f"Chemin recherché : {MODEL_PATH}"
    )

    print(
        "Des poids aléatoires seront utilisés."
    )


model.eval()

CKPT_INFO = {}
try:
    _c = globals().get("ckpt") or {}
    for _k, _v in {**{k: v for k, v in _c.items() if k != "args"}, **{"arg." + k: v for k, v in (_c.get("args") or {}).items()}}.items():
        if isinstance(_v, (int, float, str, bool)) and len(str(_v)) < 80:
            CKPT_INFO[_k] = _v
except Exception:
    pass


# ============================================================================
# CACHE LRU
# ============================================================================

CACHE = OrderedDict()


def cache_put(key, value):
    """
    Ajoute une entrée au cache LRU.
    """

    CACHE[key] = value

    CACHE.move_to_end(key)

    while len(CACHE) > CACHE_SIZE:

        CACHE.popitem(
            last=False
        )


# ============================================================================
# UTILITAIRES
# ============================================================================

def save(img, name):
    """
    Sauvegarde une image dans uploads.
    """

    path = UPLOAD_FOLDER / name

    img.save(path)

    return name


def allowed(filename):
    """
    Vérifie l'extension.
    """

    return (
        Path(filename)
        .suffix
        .lower()
        in IMG_EXT
    )


def compute_gain(ab):
    """
    Calcule et affiche le gain automatique.

    Retourne toujours un float.
    """

    chroma = mean_chroma(ab)

    gain = auto_gain(
        ab,
        TARGET_CHROMA,
    )

    print(
        f"Chroma moyenne : "
        f"{chroma:.2f}"
    )

    print(
        f"Cible chroma : "
        f"{TARGET_CHROMA:.2f}"
    )

    print(
        f"Gain automatique : "
        f"x{gain:.2f}"
    )

    return gain


# ============================================================================
# ROUTE DE LA LANDING PAGE (INTERFACE 1)
# ============================================================================

@app.route("/", methods=["GET"])
def landing():
    """
    Sert l'interface de présentation (Landing Page) de CHROMIA AI.
    """
    return render_template(
        "landing.html",
        auth_cfg=public_config(),
    )


# ============================================================================
# PROJETS (historique par utilisateur, stocké dans users.db)
# ============================================================================

def init_projects():
    with db() as con:
        con.execute("""CREATE TABLE IF NOT EXISTS projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL,
            filename TEXT NOT NULL, output_filename TEXT NOT NULL,
            variants TEXT NOT NULL, uncertainty TEXT, infer_ms REAL,
            title TEXT, created_at REAL NOT NULL)""")
        con.execute("CREATE INDEX IF NOT EXISTS idx_projects_user ON projects(user_id, created_at)")


init_projects()


@app.context_processor
def inject_i18n():
    lang = pick_lang(request.cookies.get("lang"), request.headers.get("Accept-Language"))
    return {"t": translator(lang), "lang": lang}


@app.route("/lang/<code>")
def set_lang(code):
    """Mémorise la langue choisie puis revient à la page précédente."""
    back = request.referrer or url_for("landing")
    resp = redirect(back)
    if code in LANGS:
        resp.set_cookie("lang", code, max_age=60 * 60 * 24 * 365, samesite="Lax")
    return resp


@app.context_processor
def inject_project_count():
    u = current_user()
    if not u:
        return {}
    with db() as con:
        n = con.execute("SELECT COUNT(*) FROM projects WHERE user_id=?", (u["id"],)).fetchone()[0]
    return {"project_count": n}


def get_project(pid):
    with db() as con:
        row = con.execute("SELECT * FROM projects WHERE id=? AND user_id=?",
                          (pid, current_user()["id"])).fetchone()
    if not row:
        abort(404)
    return row


def delete_files(row):
    names = [row["filename"], row["output_filename"], *json.loads(row["variants"] or "[]")]
    if row["uncertainty"]:
        names.append(row["uncertainty"])
    root = UPLOAD_FOLDER.resolve()
    for n in names:
        p = (UPLOAD_FOLDER / n).resolve()
        if p.parent == root:
            p.unlink(missing_ok=True)
        CACHE.pop(n, None)


@app.route("/projects")
@login_required
def projects():
    with db() as con:
        rows = con.execute("SELECT * FROM projects WHERE user_id=? ORDER BY created_at DESC",
                           (current_user()["id"],)).fetchall()
    cards = [{"id": r["id"], "title": r["title"], "output": r["output_filename"], "orig": r["filename"],
              "infer_ms": r["infer_ms"] or 0,
              "date": time.strftime("%d/%m/%Y", time.localtime(r["created_at"]))}
             for r in rows if (UPLOAD_FOLDER / r["output_filename"]).exists()]
    return render_template("index.html", page="projects", projects=cards)


@app.route("/projects/<int:pid>")
@login_required
def project_view(pid):
    row = get_project(pid)
    if not (UPLOAD_FOLDER / row["filename"]).exists():
        abort(404)
    return render_template(
        "index.html", page="workspace", project_title=row["title"],
        filename=row["filename"], output_filename=row["output_filename"],
        variants=json.loads(row["variants"] or "[]"),
        uncertainty_filename=row["uncertainty"],
        infer_ms=row["infer_ms"] or 0.0, device=device, pid=pid)


def _lab(path, side=192):
    """Image sRGB -> CIELAB (D65), réduite pour rester rapide."""
    im = (path.copy() if isinstance(path, Image.Image) else Image.open(path)).convert("RGB")
    im.thumbnail((side, side))
    c = np.asarray(im, dtype=np.float64) / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = (c @ M.T) / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.dstack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])]), im


@app.route("/projects/<int:pid>/analysis")
@login_required
def project_analysis(pid):
    """Mesures objectives sur le résultat enregistré (aucune ré-inférence)."""
    row = get_project(pid)
    out_p = UPLOAD_FOLDER / row["output_filename"]
    if not out_p.exists():
        abort(404)
    lab, im = _lab(out_p)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    chroma = np.hypot(a, b)
    rgb = np.asarray(im, dtype=np.float64)
    rg, yb = rgb[..., 0] - rgb[..., 1], 0.5 * (rgb[..., 0] + rgb[..., 1]) - rgb[..., 2]
    colorfulness = float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))
    ha, ea = np.histogram(a, bins=48, range=(-100, 100))
    hb, eb = np.histogram(b, bins=48, range=(-100, 100))
    h2, _, _ = np.histogram2d(a.ravel(), b.ravel(), bins=32, range=[[-100, 100], [-100, 100]])
    res = {
        "size": list(Image.open(out_p).size),
        "L_mean": float(L.mean()), "L_std": float(L.std()),
        "chroma_mean": float(chroma.mean()), "chroma_p95": float(np.percentile(chroma, 95)),
        "a_mean": float(a.mean()), "b_mean": float(b.mean()),
        "colorfulness": colorfulness,
        "hist_a": ha.tolist(), "hist_b": hb.tolist(),
        "gamut": (np.log1p(h2) / max(np.log1p(h2).max(), 1e-9)).round(3).tolist(),
        "params": int(sum(p.numel() for p in model.parameters())),
        "device": str(device), "infer_ms": row["infer_ms"] or 0.0,
    }
    mp = Image.open(out_p).size[0] * Image.open(out_p).size[1] / 1e6
    res["mp_per_s"] = mp / max((row["infer_ms"] or 1) / 1000, 1e-6)
    # Dispersion MC-Dropout : écart-type des canaux a,b entre les variantes (incertitude épistémique)
    vs = []
    for n in json.loads(row["variants"] or "[]"):
        p = UPLOAD_FOLDER / n
        if p.exists():
            vl, _ = _lab(p)
            if vl.shape == lab.shape:
                vs.append(vl[..., 1:])
    if len(vs) >= 2:
        sd = np.sqrt(np.stack(vs).var(axis=0).sum(axis=-1))
        thr = max(2 * float(sd.mean()), 1e-6)
        res["mc"] = {"n": len(vs), "sigma_mean": float(sd.mean()), "sigma_p95": float(np.percentile(sd, 95)),
                     "high_pct": float((sd > thr).mean() * 100)}
    return jsonify(res)


# ============================================================================
# ÉVALUATION QUANTITATIVE (vérité terrain connue) ET PAGE MODÈLE
# ============================================================================

def _box(x, k):
    """Moyenne glissante k×k (fenêtre valide) par image intégrale."""
    c = np.cumsum(np.cumsum(np.pad(x, ((1, 0), (1, 0))), 0), 1)
    return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)


def ssim_gray(a, b, k=7):
    """SSIM moyen (fenêtre uniforme k×k) entre deux images en niveaux de gris 0-255."""
    a, b = a.astype(np.float64), b.astype(np.float64)
    ma, mb = _box(a, k), _box(b, k)
    va, vb, cov = _box(a * a, k) - ma ** 2, _box(b * b, k) - mb ** 2, _box(a * b, k) - ma * mb
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    return float((((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma ** 2 + mb ** 2 + c1) * (va + vb + c2))).mean())


def _scores(gt_img, pr_img):
    """PSNR, SSIM (moyenne RVB), ΔE76 entre deux images RGB de même taille."""
    g, p = np.asarray(gt_img, dtype=np.float64), np.asarray(pr_img, dtype=np.float64)
    mse = float(((g - p) ** 2).mean())
    psnr = 99.0 if mse < 1e-9 else float(10 * np.log10(255 ** 2 / mse))
    ssim = float(np.mean([ssim_gray(g[..., i], p[..., i]) for i in range(3)]))   # moyenne sur R, V, B
    lg, _ = _lab(gt_img, 4096)
    lp, _ = _lab(pr_img, 4096)
    de = np.sqrt(((lg - lp) ** 2).sum(-1))
    return {"psnr": psnr, "ssim": ssim, "de_mean": float(de.mean()), "de_p95": float(np.percentile(de, 95))}, de


def _heat(de, vmax=40.0):
    """Carte d'erreur ΔE : bleu nuit → violet → rose → jaune."""
    t = np.clip(de / vmax, 0, 1)[..., None]
    stops = np.array([[5, 7, 40], [139, 92, 246], [236, 72, 153], [250, 204, 21]], dtype=np.float64)
    pos = t * (len(stops) - 1)
    i = np.clip(np.floor(pos).astype(int)[..., 0], 0, len(stops) - 2)
    f = pos - i[..., None]
    return Image.fromarray((stops[i] * (1 - f) + stops[i + 1] * f).astype("uint8"))


@app.route("/evaluate", methods=["GET", "POST"])
@login_required
def evaluate():
    if request.method == "GET":
        return render_template("index.html", page="evaluate", ev=None, err=request.args.get("err"))
    file = request.files.get("file")
    if not file or not file.filename or not allowed(file.filename):
        return redirect(url_for("evaluate", err="invalid"))
    try:
        gt = Image.open(file.stream).convert("RGB")
    except (UnidentifiedImageError, OSError):
        return redirect(url_for("evaluate", err="invalid"))
    gt.thumbnail((768, 768))
    gray = gt.convert("L").convert("RGB")           # entrée monochrome simulée
    t0 = time.perf_counter()
    L_full, ab = predict_ab(model, gray, device, size=infer_size)
    infer_ms = (time.perf_counter() - t0) * 1000
    pred = compose_rgb(L_full, ab * compute_gain(ab)).convert("RGB")
    if pred.size != gt.size:
        pred = pred.resize(gt.size, Image.LANCZOS)
    m, de = _scores(gt, pred)
    base, _ = _scores(gt, gray)                     # référence : ne rien colorier
    tag = uuid.uuid4().hex[:8]
    names = {"gt": save(gt, f"eval_{tag}_gt.png"), "gray": save(gray, f"eval_{tag}_gray.png"),
             "pred": save(pred, f"eval_{tag}_pred.png"), "err": save(_heat(de), f"eval_{tag}_err.png")}
    return render_template("index.html", page="evaluate", err=None, ev={
        "img": names, "m": m, "base": base, "ms": infer_ms, "size": gt.size,
        "de_hist": np.histogram(de, bins=20, range=(0, 40))[0].tolist(), "de_max": 40})


@app.route("/model")
@login_required
def model_page():
    layers = []
    for n, mod in model.named_children():
        c = sum(p.numel() for p in mod.parameters())
        layers.append({"name": n, "type": type(mod).__name__, "params": c})
    total = sum(l["params"] for l in layers) or 1
    for l in layers:
        l["pct"] = round(100 * l["params"] / total, 1)
    info = {"Architecture": type(model).__name__, "Parameters": f"{sum(p.numel() for p in model.parameters()):,}",
            "Input": f"{infer_size}×{infer_size} · L channel", "Output": "a*, b* chrominance (CIELAB)",
            "Uncertainty": f"MC-Dropout · {N_VARIANTS} stochastic passes", "Device": str(device),
            "Weights": MODEL_PATH.name if MODEL_PATH.exists() else "none (random init)"}
    return render_template("index.html", page="model", layers=layers, info=info, ckpt=CKPT_INFO,
                           trained=MODEL_PATH.exists())


@app.route("/projects/<int:pid>/rename", methods=["POST"])
@login_required
def project_rename(pid):
    get_project(pid)
    title = " ".join(str((request.get_json(silent=True) or {}).get("title", "")).split())[:60]
    if not title:
        return jsonify(error="Please enter a name."), 400
    with db() as con:
        con.execute("UPDATE projects SET title=? WHERE id=?", (title, pid))
    return jsonify(ok=True, title=title)


@app.route("/projects/<int:pid>/delete", methods=["POST"])
@login_required
def project_delete(pid):
    row = get_project(pid)
    delete_files(row)
    with db() as con:
        con.execute("DELETE FROM projects WHERE id=?", (pid,))
    return redirect(url_for("projects"))


@app.route("/projects/clear", methods=["POST"])
@login_required
def projects_clear():
    with db() as con:
        rows = con.execute("SELECT * FROM projects WHERE user_id=?", (current_user()["id"],)).fetchall()
        con.execute("DELETE FROM projects WHERE user_id=?", (current_user()["id"],))
    for r in rows:
        delete_files(r)
    return redirect(url_for("projects"))


@app.route("/settings")
@login_required
def settings():
    return render_template("index.html", page="settings")


# ============================================================================
# ROUTE PRINCIPALE DU DASHBOARD (INTERFACE 2)
# ============================================================================

@app.route(
    "/dashboard",
    methods=["GET", "POST"],
)
@login_required
def index():

    # ------------------------------------------------------------------------
    # GET
    # ------------------------------------------------------------------------

    if request.method == "GET":

        with db() as con:
            rows = con.execute("SELECT * FROM projects WHERE user_id=? ORDER BY created_at DESC LIMIT 12",
                               (current_user()["id"],)).fetchall()
        recent = [{"id": r["id"], "title": r["title"], "output": r["output_filename"]}
                  for r in rows if (UPLOAD_FOLDER / r["output_filename"]).exists()][:4]
        return render_template(
            "index.html",
            page="workspace",
            err=request.args.get("err"),
            recent=recent,
        )

    # ------------------------------------------------------------------------
    # POST
    # ------------------------------------------------------------------------

    file = request.files.get(
        "file"
    )

    # ------------------------------------------------------------------------
    # Validation du fichier
    # ------------------------------------------------------------------------

    if (
        not file
        or not file.filename
        or not allowed(file.filename)
    ):
        return redirect(url_for("index", err="invalid"))

    # ------------------------------------------------------------------------
    # Nom sécurisé
    # ------------------------------------------------------------------------

    safe_name = secure_filename(
        file.filename
    )

    filename = (
        f"{uuid.uuid4().hex[:8]}_"
        f"{safe_name}"
    )

    input_path = (
        UPLOAD_FOLDER
        / filename
    )

    # ------------------------------------------------------------------------
    # Sauvegarde upload
    # ------------------------------------------------------------------------

    file.save(input_path)

    # ------------------------------------------------------------------------
    # Ouverture image
    # ------------------------------------------------------------------------

    try:

        pil_img = (
            Image.open(input_path)
            .convert("RGB")
        )

    except (
        UnidentifiedImageError,
        OSError,
    ):

        input_path.unlink(
            missing_ok=True
        )

        return redirect(url_for("index", err="invalid"))

    # ------------------------------------------------------------------------
    # Inférence principale
    # ------------------------------------------------------------------------

    t0 = time.perf_counter()

    L_full, ab = predict_ab(
        model,
        pil_img,
        device,
        size=infer_size,
    )

    infer_ms = (
        time.perf_counter()
        - t0
    ) * 1000

    # ------------------------------------------------------------------------
    # Gain automatique
    # ------------------------------------------------------------------------

    gain = compute_gain(ab)

    # Applique le gain.
    ab_adjusted = ab * gain

    # ------------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------------

    cache_put(
        filename,
        (
            L_full,
            ab_adjusted,
        ),
    )

    # ------------------------------------------------------------------------
    # Image principale
    # ------------------------------------------------------------------------

    output_filename = save(
        compose_rgb(
            L_full,
            ab_adjusted,
        ),
        "colorized_" + filename,
    )

    # ------------------------------------------------------------------------
    # MC-Dropout
    # ------------------------------------------------------------------------

    _, samples = predict_ab(
        model,
        pil_img,
        device,
        size=infer_size,
        n_samples=N_VARIANTS,
    )

    # ------------------------------------------------------------------------
    # Variantes
    # ------------------------------------------------------------------------

    stem = Path(
        filename
    ).stem

    variants = []

    for k in range(
        N_VARIANTS
    ):

        variant_ab = (
            samples[k] * gain
        )

        variant_img = compose_rgb(
            L_full,
            variant_ab,
        )

        variant_name = (
            f"var{k}_{stem}.png"
        )

        variants.append(
            save(
                variant_img,
                variant_name,
            )
        )

    # ------------------------------------------------------------------------
    # Carte d'incertitude
    # ------------------------------------------------------------------------

    uncertainty_name = (
        f"unc_{stem}.png"
    )

    unc_name = save(
        uncertainty_map(samples),
        uncertainty_name,
    )

    # ------------------------------------------------------------------------
    # Logs
    # ------------------------------------------------------------------------

    print(
        f"[{filename}] "
        f"inférence : "
        f"{infer_ms:.1f} ms"
    )

    print(
        f"[{filename}] "
        f"variantes MC-Dropout : "
        f"{N_VARIANTS}"
    )

    # ------------------------------------------------------------------------
    # Enregistrement du projet puis affichage du résultat
    # ------------------------------------------------------------------------

    with db() as con:
        cur = con.execute(
            "INSERT INTO projects(user_id,filename,output_filename,variants,uncertainty,"
            "infer_ms,title,created_at) VALUES(?,?,?,?,?,?,?,?)",
            (current_user()["id"], filename, output_filename, json.dumps(variants),
             unc_name, infer_ms, safe_name or "image", time.time()))
        pid = cur.lastrowid

    return redirect(url_for("project_view", pid=pid))


# ============================================================================
# AJUSTEMENT INTERACTIF
# ============================================================================

@app.route(
    "/adjust/<filename>"
)
@login_required
def adjust(filename):
    """
    Recolorise une image à la volée.

    Aucun nouvel appel au réseau si l'image
    est déjà dans le cache.

    Paramètres :

        sat
            saturation
            [0, 3]

        temp
            température
            [-40, 40]
    """

    # ------------------------------------------------------------------------
    # Paramètres
    # ------------------------------------------------------------------------

    try:

        sat = float(
            request.args.get(
                "sat",
                1,
            )
        )

        temp = float(
            request.args.get(
                "temp",
                0,
            )
        )

    except ValueError:

        abort(400)

    # ------------------------------------------------------------------------
    # Limitation
    # ------------------------------------------------------------------------

    sat = min(
        max(sat, 0.0),
        3.0,
    )

    temp = min(
        max(temp, -40.0),
        40.0,
    )

    # ------------------------------------------------------------------------
    # Sécurisation nom
    # ------------------------------------------------------------------------

    filename = secure_filename(
        filename
    )

    # ------------------------------------------------------------------------
    # Récupération cache
    # ------------------------------------------------------------------------

    if filename not in CACHE:

        path = (
            UPLOAD_FOLDER
            / filename
        )

        if not path.exists():

            abort(404)

        try:

            pil_img = (
                Image.open(path)
                .convert("RGB")
            )

        except (
            UnidentifiedImageError,
            OSError,
        ):

            abort(400)

        # --------------------------------------------------------------
        # Nouvelle inférence
        # --------------------------------------------------------------

        L0, ab0 = predict_ab(
            model,
            pil_img,
            device,
            size=infer_size,
        )

        # --------------------------------------------------------------
        # Gain automatique
        # --------------------------------------------------------------

        gain = compute_gain(
            ab0
        )

        ab0 = ab0 * gain

        # --------------------------------------------------------------
        # Cache
        # --------------------------------------------------------------

        cache_put(
            filename,
            (
                L0,
                ab0,
            ),
        )

    # ------------------------------------------------------------------------
    # Mettre l'entrée en tête du cache
    # ------------------------------------------------------------------------

    CACHE.move_to_end(
        filename
    )

    L_full, ab = CACHE[
        filename
    ]

    # ------------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------------

    result = compose_rgb(
        L_full,
        ab,
        saturation=sat,
        temperature=temp,
    )

    # ------------------------------------------------------------------------
    # Buffer mémoire
    # ------------------------------------------------------------------------

    buf = io.BytesIO()

    if request.args.get("fmt") == "png":
        result.save(buf, format="PNG")
        mime = "image/png"
    else:
        result.save(buf, format="JPEG", quality=92)
        mime = "image/jpeg"

    buf.seek(0)

    return send_file(
        buf,
        mimetype=mime,
    )


# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.route(
    "/health"
)
def health():
    """
    Endpoint de contrôle pour Docker,
    supervision ou tests.
    """

    return jsonify(
        status="ok",
        device=device,
        model_loaded=MODEL_PATH.exists(),
        model_path=str(MODEL_PATH),
        input_size=infer_size,
        target_chroma=TARGET_CHROMA,
        cache_size=len(CACHE),
    )


# ============================================================================
# ERREUR 413
# ============================================================================

@app.errorhandler(413)
def too_large(_):
    return (
        f"Fichier trop volumineux "
        f"(max {MAX_UPLOAD_MB} Mo).",
        413,
    )


# ============================================================================
# LANCEMENT
# ============================================================================

if __name__ == "__main__":

    debug = (
        os.environ.get(
            "COLORVISION_DEBUG",
            "0",
        )
        == "1"
    )

    print()
    print("=" * 60)
    print("CHROMIA AI")
    print("=" * 60)
    print(
        f"Device          : {device}"
    )
    print(
        f"Modèle          : {MODEL_PATH}"
    )
    print(
        f"Input réseau    : "
        f"{infer_size}x{infer_size}"
    )
    print(
        f"Target chroma   : "
        f"{TARGET_CHROMA}"
    )
    print(
        f"MC-Dropout      : "
        f"{N_VARIANTS} variantes"
    )
    print(
        f"Debug           : "
        f"{debug}"
    )
    print("=" * 60)
    print()

    app.run(
        debug=debug
    )