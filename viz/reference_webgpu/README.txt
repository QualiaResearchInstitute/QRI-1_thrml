Replication Realtime WebGPU Skeleton — Integration Notes
========================================================

Source
------
- `reference_webgpu/ReplicationRealtimeWebGPUSkeleton.jsx` — full React/WebGPU example.

Key Sections
------------
- `physicsWGSL`: Kuramoto-style compute kernel running on WebGPU with FP16 storage.
- `jfaInitWGSL` & `jfaPassWGSL`: optional Jump-Flood scaffolding for distance fields.
- React component: handles context setup, camera/image uploads, simulation tick loop, and minimal UI.
- Utility helpers: resolution picking, cover/contain mapping, float16 converter.

Recommended Usage Inside `oscilleditor`
----------------------------------------
1. Treat this file as a reference while porting WebGPU ideas into the existing WebGL pipeline.
2. If/when we add a WebGPU mode, reuse the compute kernels (`physicsWGSL`, JFA) and the resize/upload helpers.
3. The React wrapper demonstrates state management, but `oscilleditor` can adapt the logic into vanilla JS modules.
4. Keep the copy in this folder so future contributors have a self-contained example of a modern WebGPU setup.

How to view updates
-------------------
- Open the file directly in the editor for reference.
- If additional helper files are needed later, place them beside this README.


