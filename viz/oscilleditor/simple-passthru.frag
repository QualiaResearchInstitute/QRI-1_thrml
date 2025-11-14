#version 300 es

precision mediump float;

uniform sampler2D uTexture;

in vec2 vTexCoord; // in webgl2, 'varying' becomes 'out' in vertex shader and 'in' in fragment shader
out vec4 fragColor;  // in webgl2, the builtin gl_FragColor no longer exists, so define an output variable for the fragment color

void main() 
{
	fragColor = texture(uTexture, vTexCoord);
}
