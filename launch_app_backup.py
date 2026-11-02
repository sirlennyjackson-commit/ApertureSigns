import sys
import os
import subprocess
import threading
import json

from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtWebEngineWidgets import QWebEngineView

import pyttsx3


# ============================================================
# APERTURE SIGNS — LAUNCH ENGINE
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRACKER = os.path.join(
    BASE_DIR,
    "backend",
    "vision",
    "hand_tracker.py"
)


# ============================================================
# TTS WORKER SIGNALS
# ============================================================

class WorkerSignals(QObject):
    audio_finished = pyqtSignal()


# ============================================================
# MAIN UI
# ============================================================

HTML = r"""
<!DOCTYPE html>
<html>
<head>

<meta charset="UTF-8">

<style>

* {
    box-sizing: border-box;
}

html,
body {
    width: 100%;
    height: 100%;
    margin: 0;
    overflow: hidden;
}

body {
    background:
        radial-gradient(
            circle at 72% 42%,
            rgba(0,229,255,0.08),
            transparent 28%
        ),
        radial-gradient(
            circle at 15% 80%,
            rgba(168,255,0,0.05),
            transparent 25%
        ),
        #05080d;

    color: white;

    font-family:
        "Segoe UI",
        "Inter",
        Arial,
        sans-serif;
}


/* =========================================================
   GLOBAL
   ========================================================= */

.screen {
    position: absolute;

    inset: 0;

    width: 100%;
    height: 100%;

    opacity: 0;

    pointer-events: none;

    transition:
        opacity 650ms ease,
        transform 650ms cubic-bezier(.2,.8,.2,1);

    transform: scale(1.025);
}

.screen.active {
    opacity: 1;
    pointer-events: auto;
    transform: scale(1);
}


/* =========================================================
   SCREEN 1 — LAUNCH
   ========================================================= */

#launch {

    padding: 38px 46px;

    display: flex;
    flex-direction: column;

    background:
        linear-gradient(
            115deg,
            rgba(255,255,255,0.018),
            transparent 40%
        ),
        radial-gradient(
            circle at 80% 50%,
            rgba(0,229,255,0.06),
            transparent 32%
        );
}


/* TOP BAR */

.topbar {

    display: flex;

    justify-content: space-between;

    align-items: center;

    height: 42px;
}

.brand {

    display: flex;

    align-items: center;

    gap: 12px;

    font-size: 11px;

    font-weight: 700;

    letter-spacing: 3px;

    color: #eafcff;
}

.brand-mark {

    width: 25px;
    height: 25px;

    border: 2px solid #00e5ff;

    border-radius: 50%;

    position: relative;

    box-shadow:
        0 0 15px rgba(0,229,255,.25);
}

.brand-mark::before,
.brand-mark::after {

    content: "";

    position: absolute;

    left: 50%;
    top: 50%;

    transform: translate(-50%, -50%);

    border-radius: 50%;
}

.brand-mark::before {

    width: 12px;
    height: 12px;

    border: 2px solid #a8ff00;
}

.brand-mark::after {

    width: 4px;
    height: 4px;

    background: #a8ff00;

    box-shadow:
        0 0 10px #a8ff00;
}

.status {

    display: flex;

    align-items: center;

    gap: 9px;

    color: #697783;

    font-size: 9px;

    letter-spacing: 2px;
}

.status-dot {

    width: 6px;
    height: 6px;

    border-radius: 50%;

    background: #a8ff00;

    box-shadow:
        0 0 12px #a8ff00;
}


/* MAIN */

.launch-main {

    flex: 1;

    display: flex;

    align-items: center;

    justify-content: space-between;

    padding: 0 35px 0 10px;
}


/* LEFT */

.launch-copy {

    width: 54%;

    position: relative;

    z-index: 3;
}

.eyebrow {

    font-size: 10px;

    color: #00e5ff;

    letter-spacing: 4px;

    font-weight: 700;

    margin-bottom: 22px;
}

.title {

    font-size: 67px;

    line-height: .91;

    font-weight: 800;

    letter-spacing: -4px;

    margin: 0;
}

.title span {

    color: #a8ff00;

    text-shadow:
        0 0 25px rgba(168,255,0,.16);
}

.subtitle {

    max-width: 520px;

    color: #71808d;

    font-size: 13px;

    line-height: 1.7;

    margin-top: 24px;

    letter-spacing: .4px;
}


/* SYSTEM TAG */

.system-tag {

    display: inline-flex;

    align-items: center;

    gap: 9px;

    margin-top: 25px;

    padding: 9px 13px;

    border: 1px solid rgba(0,229,255,.16);

    background: rgba(0,229,255,.025);

    color: #84939e;

    font-size: 9px;

    letter-spacing: 1.5px;
}

.system-tag i {

    width: 5px;
    height: 5px;

    border-radius: 50%;

    background: #00e5ff;

    box-shadow: 0 0 9px #00e5ff;
}


/* ORBIT */

.launch-visual {

    width: 42%;

    height: 380px;

    position: relative;

    display: flex;

    justify-content: center;

    align-items: center;
}

.orbit {

    position: absolute;

    border-radius: 50%;

    border: 1px solid rgba(0,229,255,.16);
}

.orbit.one {

    width: 330px;
    height: 330px;

    animation: rotate 16s linear infinite;

    border-top-color: #00e5ff;
    border-bottom-color: transparent;
}

.orbit.two {

    width: 245px;
    height: 245px;

    animation: rotateReverse 10s linear infinite;

    border-left-color: #a8ff00;
    border-right-color: transparent;
}

.orbit.three {

    width: 155px;
    height: 155px;

    border-color: rgba(255,255,255,.06);
}

.core {

    width: 43px;
    height: 43px;

    border-radius: 50%;

    background:
        radial-gradient(
            circle,
            #dfff99 0%,
            #a8ff00 28%,
            #273900 65%,
            transparent 70%
        );

    box-shadow:
        0 0 25px rgba(168,255,0,.45),
        0 0 80px rgba(168,255,0,.12);
}


/* ORBIT POINTS */

.node {

    position: absolute;

    width: 7px;
    height: 7px;

    border-radius: 50%;

    background: #00e5ff;

    box-shadow:
        0 0 13px #00e5ff;
}

.node.a {
    top: 20%;
    left: 26%;
}

.node.b {
    bottom: 17%;
    right: 17%;

    background: #a8ff00;

    box-shadow:
        0 0 13px #a8ff00;
}

.node.c {
    top: 48%;
    right: 3%;
}


/* PROGRESS */

.launch-footer {

    padding: 0 8px 4px;
}

.progress-info {

    display: flex;

    justify-content: space-between;

    align-items: center;

    font-family:
        Consolas,
        monospace;

    font-size: 9px;

    letter-spacing: 1.8px;

    color: #65737e;

    margin-bottom: 9px;
}

.progress-percent {

    color: #a8ff00;

    font-size: 11px;
}

.progress-track {

    width: 100%;

    height: 3px;

    background: rgba(255,255,255,.06);

    overflow: hidden;
}

.progress-fill {

    width: 0%;

    height: 100%;

    background:
        linear-gradient(
            90deg,
            #00e5ff,
            #a8ff00
        );

    box-shadow:
        0 0 14px rgba(168,255,0,.5);

    transition: width 80ms linear;
}


/* =========================================================
   APERTURE SIGNS — ENGINEERING TRANSITION
   ========================================================= */

#handoff {

    display: flex;
    align-items: center;
    justify-content: center;

    background:

        linear-gradient(
            115deg,
            rgba(0,0,0,.45),
            transparent 55%
        ),

        radial-gradient(
            ellipse at 20% 55%,
            rgba(0,229,255,.12),
            transparent 35%
        ),

        radial-gradient(
            ellipse at 85% 30%,
            rgba(168,255,0,.07),
            transparent 30%
        ),

        #070b11;

    overflow: hidden;
}


/* ---------------------------------------------------------
   ENGINEERING BACKGROUND
   --------------------------------------------------------- */

#handoff::before {

    content: "";

    position: absolute;

    width: 150%;
    height: 180px;

    left: -25%;
    top: 40%;

    transform: rotate(-7deg);

    background:

        linear-gradient(
            90deg,
            transparent,
            rgba(0,229,255,.12),
            rgba(168,255,0,.08),
            transparent
        );

    filter: blur(3px);

    animation:
        engineeringSweep 5s linear infinite;

    pointer-events: none;
}


#handoff::after {

    content: "";

    position: absolute;

    inset: 0;

    background-image:

        linear-gradient(
            rgba(255,255,255,.018) 1px,
            transparent 1px
        ),

        linear-gradient(
            90deg,
            rgba(255,255,255,.018) 1px,
            transparent 1px
        );

    background-size:
        42px 42px;

    mask-image:
        linear-gradient(
            to bottom,
            transparent,
            black 25%,
            black 75%,
            transparent
        );

    pointer-events: none;
}


/* ---------------------------------------------------------
   MAIN ENGINEERING FRAME
   --------------------------------------------------------- */

.handoff-wrapper {

    position: relative;

    width: 900px;

    z-index: 5;
}


/* ---------------------------------------------------------
   TOP TECHNICAL BAR
   --------------------------------------------------------- */

.engineering-top {

    display: flex;

    align-items: center;

    justify-content: space-between;

    margin-bottom: 10px;

    font-family:
        Consolas,
        monospace;

    font-size: 8px;

    letter-spacing: 2px;

    color: #56636e;
}

.engineering-top strong {

    color: #00e5ff;

    font-weight: 600;
}

.live-indicator {

    display: flex;

    align-items: center;

    gap: 7px;
}

.live-dot {

    width: 5px;
    height: 5px;

    border-radius: 50%;

    background: #a8ff00;

    box-shadow:
        0 0 10px #a8ff00;
}


/* ---------------------------------------------------------
   MAIN CARD
   --------------------------------------------------------- */

.handoff-card {

    position: relative;

    width: 100%;

    min-height: 390px;

    overflow: hidden;

    padding: 34px 38px;

    border:
        1px solid rgba(255,255,255,.13);

    background:

        linear-gradient(
            135deg,
            rgba(255,255,255,.075),
            rgba(255,255,255,.018) 48%,
            rgba(0,0,0,.32)
        );

    box-shadow:

        0 35px 100px rgba(0,0,0,.55),

        inset 0 1px 0
        rgba(255,255,255,.07);
}


/* ---------------------------------------------------------
   F1-STYLE ANGULAR LIGHT
   --------------------------------------------------------- */

.handoff-card::before {

    content: "";

    position: absolute;

    width: 620px;
    height: 90px;

    right: -150px;
    top: 40px;

    transform:
        rotate(-12deg);

    background:

        linear-gradient(
            90deg,
            transparent,
            rgba(0,229,255,.13),
            rgba(255,255,255,.04),
            transparent
        );

    filter: blur(1px);
}


/* ---------------------------------------------------------
   HEADER & TEXT
   --------------------------------------------------------- */

.handoff-title {

    font-size: 29px;

    font-weight: 700;

    letter-spacing: -1px;

    margin-bottom: 7px;
}

.handoff-description {

    color: #687782;

    font-size: 11px;

    letter-spacing: .5px;

    margin-bottom: 32px;
}


/* ---------------------------------------------------------
   MODULES
   --------------------------------------------------------- */

.modules {

    display: grid;

    grid-template-columns:
        repeat(3, 1fr);

    gap: 12px;
}

.module {

    padding: 20px;

    border:
        1px solid rgba(255,255,255,.07);

    background:
        rgba(0,0,0,.16);

    min-height: 105px;

    transition:
        border-color .3s ease,
        background .3s ease;
}

.module.ready {

    border-color:
        rgba(168,255,0,.20);

    background:
        rgba(168,255,0,.025);
}

.module-icon {

    width: 28px;
    height: 28px;

    border:
        1px solid rgba(0,229,255,.35);

    display: flex;

    align-items: center;
    justify-content: center;

    color: #00e5ff;

    font-size: 11px;

    margin-bottom: 14px;
}

.module.ready .module-icon {

    color: #a8ff00;

    border-color:
        rgba(168,255,0,.35);
}

.module-name {

    font-size: 10px;

    font-weight: 700;

    letter-spacing: 1px;

    color: #c8d0d5;
}

.module-status {

    margin-top: 6px;

    color: #4f5c65;

    font-size: 8px;

    letter-spacing: 1.5px;
}

.module.ready .module-status {

    color: #a8ff00;
}


/* ---------------------------------------------------------
   HANDOFF FOOTER
   --------------------------------------------------------- */

.handoff-bottom {

    position: absolute;

    bottom: 0;
    left: 0;

    width: 100%;

    padding: 15px 38px;

    border-top:
        1px solid rgba(255,255,255,.06);

    display: flex;

    justify-content: space-between;

    align-items: center;
}

.handoff-progress {

    width: 250px;

    height: 2px;

    background: rgba(255,255,255,.07);
}

.handoff-progress-fill {

    width: 0%;

    height: 100%;

    background: #00e5ff;

    box-shadow: 0 0 10px #00e5ff;

    transition: width .15s linear;
}

.enter-text {

    color: #73818b;

    font-size: 9px;

    letter-spacing: 2px;
}


/* =========================================================
   SCREEN 3 — BLANK TRANSITION
   ========================================================= */

#blank {

    background: #020406;
}


/* =========================================================
   ANIMATIONS
   ========================================================= */

@keyframes rotate {

    from {
        transform: rotate(0deg);
    }

    to {
        transform: rotate(360deg);
    }
}

@keyframes rotateReverse {

    from {
        transform: rotate(360deg);
    }

    to {
        transform: rotate(0deg);
    }
}

@keyframes engineeringSweep {

    0% {
        transform: rotate(-7deg) translateY(-100px);
        opacity: 0;
    }

    50% {
        opacity: 1;
    }

    100% {
        transform: rotate(-7deg) translateY(100px);
        opacity: 0;
    }
}


/* =========================================================
   RESPONSIVE
   ========================================================= */

@media(max-width: 850px) {

    .title {
        font-size: 50px;
    }

    .launch-main {
        padding-right: 0;
    }

    .launch-visual {
        transform: scale(.8);
    }

    .handoff-wrapper {
        width: 95%;
    }

}

</style>

</head>


<body>


<!-- =======================================================
     SCREEN 1
======================================================= -->

<section id="launch" class="screen active">

    <div class="topbar">

        <div class="brand">

            <div class="brand-mark"></div>

            APERTURE SIGNS

        </div>


        <div class="status">

            <span class="status-dot"></span>

            LOCAL SESSION / HARDWARE SYNC

        </div>

    </div>


    <div class="launch-main">


        <div class="launch-copy">

            <div class="eyebrow">
                APERTURE MOTION ENGINE / AE-1
            </div>


            <h1 class="title">

                APERTURE<br>

                <span>SIGNS AE-1</span>

            </h1>


            <div
                class="subtitle"
                id="narrative"
            >
                Initializing spatial optical drivers
                and bio-adaptive tracking engine...
            </div>


            <div class="system-tag">

                <i></i>

                VISION PIPELINE INITIALIZATION

            </div>

        </div>


        <div class="launch-visual">

            <div class="orbit one"></div>

            <div class="orbit two"></div>

            <div class="orbit three"></div>

            <div class="core"></div>

            <div class="node a"></div>

            <div class="node b"></div>

            <div class="node c"></div>

        </div>


    </div>


    <div class="launch-footer">

        <div class="progress-info">

            <span id="stage">
                SYNCHRONIZING VISION PIPELINE...
            </span>

            <span
                id="pct"
                class="progress-percent"
            >
                0%
            </span>

        </div>


        <div class="progress-track">

            <div
                id="fill"
                class="progress-fill"
            ></div>

        </div>

    </div>

</section>



<!-- =======================================================
     SCREEN 2 — HANDOFF
======================================================= -->

<section id="handoff" class="screen">


    <div class="handoff-wrapper">

        <div class="engineering-top">
            <div>SYSTEM PIPELINE: <strong>AE1://PRIMARY_PIPELINE</strong></div>
            <div class="live-indicator">
                <span class="live-dot"></span>
                SYSTEM HANDOFF LIVE
            </div>
        </div>


        <div class="handoff-card">


            <div class="handoff-title">
                Vision environment ready.
            </div>


            <div class="handoff-description">
                Transferring control from the Aperture
                launch engine to the primary collection
                environment.
            </div>


            <div class="modules">


                <div
                    id="moduleVision"
                    class="module"
                >

                    <div class="module-icon">
                        ◉
                    </div>

                    <div class="module-name">
                        VISION ENGINE
                    </div>

                    <div
                        id="statusVision"
                        class="module-status"
                    >
                        SYNCHRONIZING
                    </div>

                </div>



                <div
                    id="moduleLandmark"
                    class="module"
                >

                    <div class="module-icon">
                        +
                    </div>

                    <div class="module-name">
                        LANDMARK ENGINE
                    </div>

                    <div
                        id="statusLandmark"
                        class="module-status"
                    >
                        WAITING
                    </div>

                </div>



                <div
                    id="moduleData"
                    class="module"
                >

                    <div class="module-icon">
                        □
                    </div>

                    <div class="module-name">
                        DATA PIPELINE
                    </div>

                    <div
                        id="statusData"
                        class="module-status"
                    >
                        WAITING
                    </div>

                </div>


            </div>


            <div class="handoff-bottom">

                <div class="handoff-progress">

                    <div
                        id="handoffFill"
                        class="handoff-progress-fill"
                    ></div>

                </div>


                <div
                    id="enterText"
                    class="enter-text"
                >
                    PREPARING DATA COLLECTION
                </div>

            </div>


        </div>

    </div>

</section>



<!-- =======================================================
     SCREEN 3 — CLEAN TRANSITION
======================================================= -->

<section
    id="blank"
    class="screen"
>
</section>



<script>


/* =========================================================
   BASIC HELPERS
========================================================= */

function showScreen(id) {

    document
        .querySelectorAll(".screen")
        .forEach(
            screen => screen.classList.remove("active")
        );

    const target =
        document.getElementById(id);

    if (target) {
        target.classList.add("active");
    }
}


function updateProgress(
    p,
    stage,
    narrative
) {

    const fill =
        document.getElementById("fill");

    const pct =
        document.getElementById("pct");

    const stageEl =
        document.getElementById("stage");

    const narrativeEl =
        document.getElementById("narrative");


    fill.style.width = p + "%";

    pct.innerText =
        Math.round(p) + "%";


    if (stage) {

        stageEl.innerText =
            stage;

    }


    if (narrative) {

        narrativeEl.innerText =
            narrative;

    }

}


/* =========================================================
   HANDOFF
========================================================= */

function updateHandoff(progress) {

    const fill =
        document.getElementById("handoffFill");

    fill.style.width =
        progress + "%";


    /*
       VISION ENGINE
    */

    if (progress >= 20) {

        document
            .getElementById("moduleVision")
            .classList.add("ready");

        document
            .getElementById("statusVision")
            .innerText = "READY";

    }


    /*
       LANDMARK ENGINE
    */

    if (progress >= 48) {

        document
            .getElementById("moduleLandmark")
            .classList.add("ready");

        document
            .getElementById("statusLandmark")
            .innerText = "READY";

    }


    /*
       DATA PIPELINE
    */

    if (progress >= 76) {

        document
            .getElementById("moduleData")
            .classList.add("ready");

        document
            .getElementById("statusData")
            .innerText = "READY";

    }


    if (progress >= 95) {

        document
            .getElementById("enterText")
            .innerText =
                "ENTERING DATA COLLECTION";

    }

}


/* =========================================================
   EXTERNAL PYTHON CAN STILL CALL THIS
========================================================= */

window.updateProgress =
    updateProgress;

window.showScreen =
    showScreen;

window.updateHandoff =
    updateHandoff;


</script>


</body>
</html>
"""


# ============================================================
# PYTHON LAUNCH WINDOW
# ============================================================

class LaunchWindow(QMainWindow):

    def __init__(self):

        super().__init__()

        self.setWindowTitle(
            "Aperture Signs Launch Engine"
        )

        self.setFixedSize(
            1024,
            600
        )

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
        )


        # ----------------------------------------------------
        # WEB ENGINE
        # ----------------------------------------------------

        self.browser = QWebEngineView(self)

        self.browser.setHtml(
            HTML
        )

        self.setCentralWidget(
            self.browser
        )


        # ----------------------------------------------------
        # STATE & SIGNALS
        # ----------------------------------------------------

        self.progress = 0.0

        self.handoff_progress = 0.0

        self.audio_finished = False

        self.launch_complete = False

        self.handoff_started = False

        self.blank_started = False

        self.signals = WorkerSignals()
        self.signals.audio_finished.connect(self._on_audio_finished)


        # ----------------------------------------------------
        # VOICE
        # ----------------------------------------------------

        threading.Thread(
            target=self.speak_intro,
            daemon=True
        ).start()


        # ----------------------------------------------------
        # MAIN TIMER
        # ----------------------------------------------------

        self.timer = QTimer(self)

        self.timer.timeout.connect(
            self.update_loop
        )

        self.timer.start(50)


    # ========================================================
    # VOICE & SIGNALS
    # ========================================================

    def _on_audio_finished(self):

        self.audio_finished = True


    def speak_intro(self):

        pythoncom_loaded = False

        try:

            import pythoncom

            pythoncom.CoInitialize()

            pythoncom_loaded = True

            engine = pyttsx3.init()

            engine.setProperty(
                "rate",
                160
            )

            engine.say(
                "I am engineered to break the silence "
                "between worlds. Synchronizing neural "
                "motion engine."
            )

            engine.runAndWait()

        except Exception:

            pass

        finally:

            if pythoncom_loaded:

                try:
                    import pythoncom
                    pythoncom.CoUninitialize()
                except Exception:
                    pass

            self.signals.audio_finished.emit()


    # ========================================================
    # JAVASCRIPT HELPER
    # ========================================================

    def run_js(
        self,
        function_name,
        *args
    ):

        serialized = ", ".join(
            json.dumps(arg)
            for arg in args
        )

        javascript = (
            f"{function_name}({serialized});"
        )

        self.browser.page().runJavaScript(
            javascript
        )


    # ========================================================
    # MAIN STATE MACHINE
    # ========================================================

    def update_loop(self):


        # ====================================================
        # PHASE 1
        # ORIGINAL LAUNCH PROGRESS
        # ====================================================

        if not self.launch_complete:

            if self.progress < 100:

                self.progress += 0.4


            # ----------------------------------------------
            # EXISTING STAGES
            # ----------------------------------------------

            stage = (
                "SYNCHRONIZING VISION PIPELINE..."
            )

            narrative = (
                "Initializing spatial optical drivers "
                "and bio-adaptive tracking engine..."
            )


            if self.progress > 35:

                stage = (
                    "CALIBRATING LANDMARK VECTORS..."
                )

                narrative = (
                    "Mapping 21 keypoint coordinate "
                    "vectors in real time..."
                )


            if self.progress > 75:

                stage = (
                    "LAUNCHING VISION TRACKER..."
                )

                narrative = (
                    "System initialized. Transferring "
                    "process control to primary pipeline..."
                )


            self.run_js(
                "updateProgress",
                min(self.progress, 100),
                stage,
                narrative
            )


            # ----------------------------------------------
            # LAUNCH COMPLETE
            # ----------------------------------------------

            if (
                self.progress >= 100
                and self.audio_finished
            ):

                self.launch_complete = True

                self.handoff_started = True

                self.handoff_progress = 0

                self.run_js(
                    "showScreen",
                    "handoff"
                )


            return


        # ====================================================
        # PHASE 2
        # SYSTEM HANDOFF
        # ====================================================

        if self.handoff_started:

            self.handoff_progress += 3.0

            self.run_js(
                "updateHandoff",
                min(
                    self.handoff_progress,
                    100
                )
            )


            if self.handoff_progress >= 100:

                self.handoff_started = False

                self.blank_started = True


                # ------------------------------------------
                # CLEAN BLANK TRANSITION
                # ------------------------------------------

                self.run_js(
                    "showScreen",
                    "blank"
                )


                # ------------------------------------------
                # GIVE THE BLANK SCREEN 600ms
                # ------------------------------------------

                QTimer.singleShot(
                    600,
                    self.launch_tracker
                )

            return


    # ========================================================
    # START ORIGINAL TRACKER
    # ========================================================

    def launch_tracker(self):

        if os.path.exists(TRACKER):

            try:

                subprocess.Popen(
                    [
                        sys.executable,
                        TRACKER
                    ],
                    cwd=BASE_DIR
                )

            except Exception as e:

                print(
                    "Failed to launch hand_tracker.py:",
                    e
                )

        else:

            print(
                "hand_tracker.py not found at:",
                TRACKER
            )


        # ----------------------------------------------------
        # CLOSE LAUNCH ENGINE
        # ----------------------------------------------------

        self.timer.stop()

        self.close()


# ============================================================
# APPLICATION ENTRY
# ============================================================

if __name__ == "__main__":

    app = QApplication(
        sys.argv
    )

    win = LaunchWindow()

    win.show()

    sys.exit(
        app.exec()
    )