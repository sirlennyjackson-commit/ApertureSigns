(function () {
  const VIEWERS = new WeakMap();
  const MODEL_URL = new URL("avatar/aperture_avatar.glb", document.baseURI).href;
  const BODY_SEGMENTS = [
    ["upperarm01.L", "left_shoulder", "left_elbow", 70],
    ["upperarm01.R", "right_shoulder", "right_elbow", 70],
    ["lowerarm02.L", "left_elbow", "left_wrist", 85],
    ["lowerarm02.R", "right_elbow", "right_wrist", 85],
  ];
  const MAX_NECK_ANGLE = (12 * Math.PI) / 180;
  const DIGITS = [
    { names: ["finger1-1", "finger1-2", "finger1-3"], points: [0, 1, 2, 4] },
    { names: ["metacarpal1", "finger2-1", "finger2-2", "finger2-3"], points: [0, 5, 6, 7, 8] },
    { names: ["metacarpal2", "finger3-1", "finger3-2", "finger3-3"], points: [0, 9, 10, 11, 12] },
    { names: ["metacarpal3", "finger4-1", "finger4-2", "finger4-3"], points: [0, 13, 14, 15, 16] },
    { names: ["metacarpal4", "finger5-1", "finger5-2", "finger5-3"], points: [0, 17, 18, 19, 20] },
  ];

  function normalizeBoneName(name) {
    return name.replace(/\s/g, "_").replace(/[\[\]\.:\/]/g, "");
  }

  function getBone(state, name) {
    return state.bones.get(normalizeBoneName(name));
  }

  function getModelBounds(model) {
    const bounds = new THREE.Box3().makeEmpty();
    const vertex = new THREE.Vector3();
    let hasSkinnedMesh = false;

    model.updateMatrixWorld(true);
    model.traverse((object) => {
      if (!object.isSkinnedMesh) return;
      hasSkinnedMesh = true;
      const positions = object.geometry.attributes.position;
      for (let index = 0; index < positions.count; index += 1) {
        vertex.fromBufferAttribute(positions, index);
        object.boneTransform(index, vertex);
        vertex.applyMatrix4(object.matrixWorld);
        bounds.expandByPoint(vertex);
      }
    });

    if (!hasSkinnedMesh) bounds.setFromObject(model);
    return bounds;
  }

  function createViewer(canvas) {
    if (typeof THREE === "undefined" || typeof THREE.GLTFLoader !== "function") {
      throw new Error("The local Three.js viewer files did not load.");
    }
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.1;
    renderer.setClearColor(0x000000, 0);

    const scene = new THREE.Scene();
    scene.add(new THREE.HemisphereLight(0xe5f0e5, 0x51483b, 2.2));
    const keyLight = new THREE.DirectionalLight(0xffe7cf, 3.2);
    keyLight.position.set(-3, 5, 4);
    scene.add(keyLight);
    const fillLight = new THREE.DirectionalLight(0xc3dfd1, 1.8);
    fillLight.position.set(4, 2, -3);
    scene.add(fillLight);

    const shadowCanvas = document.createElement("canvas");
    shadowCanvas.width = 128;
    shadowCanvas.height = 128;
    const shadowContext = shadowCanvas.getContext("2d");
    if (!shadowContext) throw new Error("Could not create the avatar floor shadow.");
    const shadowGradient = shadowContext.createRadialGradient(64, 64, 8, 64, 64, 64);
    shadowGradient.addColorStop(0, "rgba(25, 30, 25, 0.34)");
    shadowGradient.addColorStop(0.45, "rgba(25, 30, 25, 0.16)");
    shadowGradient.addColorStop(1, "rgba(25, 30, 25, 0)");
    shadowContext.fillStyle = shadowGradient;
    shadowContext.fillRect(0, 0, 128, 128);
    const shadow = new THREE.Mesh(
      new THREE.PlaneGeometry(0.82, 0.5),
      new THREE.MeshBasicMaterial({
        map: new THREE.CanvasTexture(shadowCanvas),
        transparent: true,
        depthWrite: false,
      }),
    );
    shadow.rotation.x = -Math.PI / 2;
    shadow.position.y = -0.006;
    scene.add(shadow);

    const camera = new THREE.PerspectiveCamera(32, 1, 0.01, 100);
    const holder = new THREE.Group();
    scene.add(holder);

    const loader = new THREE.GLTFLoader();
    const state = {
      bones: new Map(),
      canvas,
      camera,
      failed: false,
      holder,
      lastFrame: null,
      lastAppliedFrame: null,
      model: null,
      pointerX: null,
      renderer,
      scene,
      restPose: new Map(),
    };

    loader.load(
      MODEL_URL,
      (gltf) => {
        state.model = gltf.scene;
        holder.add(state.model);
        state.model.updateMatrixWorld(true);

        const bounds = getModelBounds(state.model);
        const size = bounds.getSize(new THREE.Vector3());
        if (!Number.isFinite(size.y) || size.y <= 0) {
          throw new Error("The generated avatar has invalid dimensions.");
        }
        state.model.position.y -= bounds.min.y;
        state.model.rotation.y = 0;
        shadow.position.y = state.model.position.y + bounds.min.y - 0.006;
        camera.position.set(0, size.y * 0.52, size.y * 2.65);
        camera.lookAt(0, size.y * 0.48, 0);
        camera.near = size.y / 100;
        camera.far = size.y * 100;
        camera.updateProjectionMatrix();

        state.model.traverse((object) => {
          if (!object.isBone) return;
          const name = normalizeBoneName(object.name);
          state.bones.set(name, object);
          object.updateWorldMatrix(true, false);
          const restWorldQuaternion = object.getWorldQuaternion(new THREE.Quaternion());
          const child = object.children.find((item) => item.isBone);
          let restDirection = null;
          if (child) {
            const start = object.getWorldPosition(new THREE.Vector3());
            const end = child.getWorldPosition(new THREE.Vector3());
            restDirection = end.sub(start).normalize();
          }
          state.restPose.set(name, {
            localQuaternion: object.quaternion.clone(),
            restDirection,
            restWorldQuaternion,
          });
        });
        applyFrame(state, state.lastFrame);
        render(state);
      },
      undefined,
      (error) => {
        state.failed = true;
        console.error("Could not load the generated avatar:", error);
        const status = document.getElementById("avatarStatus");
        if (status) status.textContent = "3D avatar could not be loaded. Check the local avatar asset.";
      },
    );

    canvas.addEventListener("pointerdown", (event) => {
      state.pointerX = event.clientX;
      canvas.setPointerCapture(event.pointerId);
    });
    canvas.addEventListener("pointermove", (event) => {
      if (state.pointerX === null) return;
      holder.rotation.y += (event.clientX - state.pointerX) * 0.008;
      state.pointerX = event.clientX;
      render(state);
    });
    const endDrag = () => {
      state.pointerX = null;
    };
    canvas.addEventListener("pointerup", endDrag);
    canvas.addEventListener("pointercancel", endDrag);
    canvas.addEventListener("webglcontextlost", (event) => {
      event.preventDefault();
      state.failed = true;
      console.error("The avatar viewer lost its WebGL context.");
    });
    return state;
  }

  function render(state) {
    const rect = state.canvas.getBoundingClientRect();
    if (!rect.width || !rect.height || state.failed) return;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.round(rect.width * ratio);
    const height = Math.round(rect.height * ratio);
    if (state.canvas.width !== width || state.canvas.height !== height) {
      state.renderer.setPixelRatio(ratio);
      state.renderer.setSize(rect.width, rect.height, false);
      state.camera.aspect = rect.width / rect.height;
      state.camera.updateProjectionMatrix();
    }
    state.renderer.render(state.scene, state.camera);
  }

  function avatarDirection(start, end) {
    return new THREE.Vector3(
      end[0] - start[0],
      start[1] - end[1],
      start[2] - end[2],
    );
  }

  function aimBone(state, name, start, end, maxAngle) {
    const normalizedName = normalizeBoneName(name);
    const bone = getBone(state, name);
    const rest = state.restPose.get(normalizedName);
    if (!bone || !rest || !rest.restDirection) return;

    const direction = avatarDirection(start, end);
    if (direction.lengthSq() < 1e-8) return;
    direction.normalize();

    const delta = new THREE.Quaternion().setFromUnitVectors(rest.restDirection, direction);
    const angle = delta.angleTo(new THREE.Quaternion());
    if (angle > maxAngle) {
      delta.slerp(new THREE.Quaternion(), 1 - maxAngle / angle);
    }
    const worldQuaternion = delta.multiply(rest.restWorldQuaternion);
    const parentQuaternion = bone.parent
      ? bone.parent.getWorldQuaternion(new THREE.Quaternion()).invert()
      : new THREE.Quaternion();
    const targetQuaternion = parentQuaternion.multiply(worldQuaternion);
    bone.quaternion.copy(targetQuaternion);
  }

  function midpoint(first, second) {
    if (!first || !second) return null;
    return [
      (first[0] + second[0]) / 2,
      (first[1] + second[1]) / 2,
      (first[2] + second[2]) / 2,
    ];
  }

  function resetPose(state) {
    state.lastFrame = null;
    state.lastAppliedFrame = null;
    if (!state.model) return;
    for (const [name, rest] of state.restPose) {
      const bone = getBone(state, name);
      if (bone) bone.quaternion.copy(rest.localQuaternion);
    }
    state.model.updateMatrixWorld(true);
    render(state);
  }

  function applyFrame(state, frame) {
    if (!state.model) {
      state.lastFrame = frame;
      return;
    }
    if (!frame) {
      resetPose(state);
      return;
    }
    if (frame === state.lastAppliedFrame) {
      render(state);
      return;
    }

    for (const [name, rest] of state.restPose) {
      const bone = getBone(state, name);
      if (bone) bone.quaternion.copy(rest.localQuaternion);
    }
    state.model.updateMatrixWorld(true);
    state.lastFrame = frame;
    state.lastAppliedFrame = frame;

    const joints = frame.joints || {};
    const leftHip = joints.left_hip;
    const rightHip = joints.right_hip;
    const leftShoulder = joints.left_shoulder;
    const rightShoulder = joints.right_shoulder;
    const points = {
      ...joints,
      hipCenter: midpoint(leftHip, rightHip),
      shoulderCenter: midpoint(leftShoulder, rightShoulder),
    };
    for (const [name, startName, endName, maxAngleDegrees] of BODY_SEGMENTS) {
      const start = points[startName];
      const end = points[endName];
      if (!start || !end) continue;
      aimBone(state, name, start, end, THREE.MathUtils.degToRad(maxAngleDegrees));
      state.model.updateMatrixWorld(true);
    }

    const nose = joints.nose;
    const shoulderCenter = points.shoulderCenter;
    if (
      nose &&
      shoulderCenter &&
      Number.isFinite(nose[0]) &&
      Number.isFinite(nose[1]) &&
      nose[1] < shoulderCenter[1]
    ) {
      aimBone(state, "neck01", shoulderCenter, nose, MAX_NECK_ANGLE);
      state.model.updateMatrixWorld(true);
    }

    for (const [side, hand] of [["L", frame.left_hand], ["R", frame.right_hand]]) {
      if (!Array.isArray(hand) || hand.length < 21) continue;
      for (const digit of DIGITS) {
        for (let index = 0; index < digit.names.length; index += 1) {
          const start = hand[digit.points[index]];
          const end = hand[digit.points[index + 1]];
          if (start && end) {
            aimBone(state, `${digit.names[index]}.${side}`, start, end, Math.PI * 0.8);
            state.model.updateMatrixWorld(true);
          }
        }
      }
    }
    render(state);
  }

  window.ApertureAvatar = {
    update(canvas, frame) {
      try {
        let state = VIEWERS.get(canvas);
        if (!state) {
          state = createViewer(canvas);
          VIEWERS.set(canvas, state);
        }
        if (state.failed) return true;
        applyFrame(state, frame);
        return true;
      } catch (error) {
        console.error("The in-app 3D avatar viewer failed:", error);
        const status = document.getElementById("avatarStatus");
        if (status) status.textContent = "3D avatar viewer unavailable: " + error.message;
        return false;
      }
    },
    reset(canvas) {
      const state = VIEWERS.get(canvas);
      if (state) resetPose(state);
    },
    resize() {
      document.querySelectorAll("#avatarCanvas, #signingAvatarCanvas").forEach((canvas) => {
        const state = VIEWERS.get(canvas);
        if (state) render(state);
      });
    },
  };
})();
