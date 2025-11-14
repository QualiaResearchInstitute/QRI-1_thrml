#version 300 es

precision mediump float;

uniform ivec2 uOscillatorTexSize;
// note that more uniforms are #included from a separate file

in vec2 vTexCoord; // in webgl2, 'varying' becomes 'out' in vertex shader and 'in' in fragment shader
out vec4 fragColor;  // in webgl2, the builtin gl_FragColor no longer exists, so define an output variable for the fragment color


// homemade file include feature, the javascript code will paste the file in here
#include-kernel-ring-function.frag


// float modI(float a,float b) {
//     float m=a-floor((a+0.5)/b)*b;
//     return floor(m+0.5);
// }

void drawFilterKernel()
{
	// this function will be used to draw onto the main canvas, which is always high-res
	// but the oscillator texture (uOscillatorTexRead) might have lower resolution
	vec2 texWidthHeight = vec2(uOscillatorTexSize);
	vec2 smoothTexelCoord = vTexCoord * texWidthHeight;
	vec2 blockyTexelCoord = floor(smoothTexelCoord); // to index into the texture, use this instead of gl_FragCoord which indexes into the canvas
	float loopRadius = ceil(kernelMaxDistance);

	smoothTexelCoord.y = texWidthHeight.y - smoothTexelCoord.y;
	blockyTexelCoord.y = texWidthHeight.y - blockyTexelCoord.y;
	// draw a pixelated kernel in the top-left corner, to show how it will actually be applied in the oscillator texture
	if (blockyTexelCoord.x <= 2.0*loopRadius && blockyTexelCoord.y <= 2.0*loopRadius)
	{
		float dx = blockyTexelCoord.x - loopRadius;
		float dy = blockyTexelCoord.y - loopRadius;
	    float distance = sqrt(dx * dx + dy * dy);
	    float couplingstrength = influenceFunction(distance, kernelMaxDistance);
	    float gray = couplingstrength/(6.0*2.0) + 0.5; // this 6.0 is the max strength of the coupling slider in the html ui
	    fragColor = vec4(gray, gray, gray, 1.0);
	}
	// draw a smooth kernel right next to it, to more clearly illustrate the pure mathematics
	else if (blockyTexelCoord.x <= 4.0*loopRadius && smoothTexelCoord.y <= 2.0*loopRadius)
	{
		smoothTexelCoord.x += -2.0*loopRadius - 1.0;
		float dx = smoothTexelCoord.x - loopRadius;
		float dy = smoothTexelCoord.y - loopRadius;
	    float distance = sqrt(dx * dx + dy * dy);
	    float couplingstrength = influenceFunction(distance, kernelMaxDistance);
	    float gray = couplingstrength/(6.0*2.0) + 0.5; // this 6.0 is the max strength of the coupling slider in the html ui
	    fragColor = vec4(gray, gray, gray, 1.0);
	}
	else
		discard;
}

void main()
{
	drawFilterKernel();
}
