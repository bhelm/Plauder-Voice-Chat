# Third-party vendored assets

These files are bundled verbatim so the browser client needs **no build step**.
They keep their own upstream licenses (all MIT), which are compatible with this
project's GPLv3 distribution.

| Path | Package | Version | License | Source |
|------|---------|---------|---------|--------|
| `three/three.module.js`, `three/three.core.js`, `three/addons/**` | three.js | r175 | MIT | https://github.com/mrdoob/three.js |
| `three-vrm/three-vrm.module.js` | @pixiv/three-vrm | v3 | MIT | https://github.com/pixiv/three-vrm |
| `three-vrm/three-vrm-animation.module.js` | @pixiv/three-vrm-animation | v3.5.5 | MIT | https://github.com/pixiv/three-vrm |

## Avatar animations (`../anims/*.vrma`)

VRM Animation clips (Angry, Blush, Clapping, Goodbye, Jump, LookAround, Relax,
Sad, Sleepy, Surprised, Thinking) taken from
[`tk256ailab/vrm-viewer`](https://github.com/tk256ailab/vrm-viewer) — MIT License.

The `joy.vrm` avatar model itself is **not** in the repo (gitignored under
`static/models/`, like the other model assets).
