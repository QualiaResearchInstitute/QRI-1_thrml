#version 300 es

in vec3 aPosition;    // 'attribute' becomes 'in'
in vec2 aTexCoord;    // 'attribute' becomes 'in'

out vec2 vTexCoord;   // 'varying' becomes 'out' in vertex shader and 'in' in fragment shader

void main() {
   vTexCoord = aTexCoord;
   gl_Position = vec4(aPosition,1.0);
}