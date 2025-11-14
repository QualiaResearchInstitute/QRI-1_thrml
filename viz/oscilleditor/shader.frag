#version 300 es

precision mediump float;

uniform sampler2D uMainTex;         // Main image texture
uniform sampler2D uDeepDreamTex;       // Custom texture (image or video)
uniform sampler2D uDisplacementMap; // Displacement video (mask)
uniform sampler2D uBlendingPattern; // blending pattern video (mask)
uniform sampler2D uOscillatorTexRead[2]; // our simulation, from oscillate.frag
uniform int uNumOscillatorTextures; // currently only support for 1 or 2
uniform sampler2D uDomainMaskTex;
uniform bool useDomains;
uniform int numDomains;
uniform float domainOrderParameter[8];
uniform float domainChaosParameter[8];

uniform float driftMaxDisplacementHoriz;
uniform float driftMaxDisplacementVert; // #TODO: join horiz & vert into a vec2?
uniform float driftPatternScale;
uniform float blendPatternScale;
uniform float deepDreamOpacity; // between 0 and 1
uniform float deepLuminosityBlendOpacity; // between 0 and 1
uniform float oscillatorOpacity; // between 0 and 1
uniform float crossFieldBlendingBalance; // for colors, conceptually mix(uOscillatorTexRead[0], uOscillatorTexRead[1], crossFieldBlendingBalance)
uniform int oscillatorColors;
uniform bool logPolarTransform[2];
uniform float corticalDepressionIntensity;

// just for debugging:
uniform bool debugDrawKernelSize;
uniform int debugDrawKernelSizeForField;
uniform float kernelMaxDistance;
uniform bool debugDrawFrequency;
uniform int frameIndex;
uniform float frequencyRange[2]; // the lowest and highest frequency that _can_ be set by the GUI sliders. not the values that the user has currently chosen! opposite of init-oscillators.frag
uniform float deltaTime; // in seconds. typically 1/60 = 0.01666667

in vec2 vTexCoord; // in webgl2, 'varying' becomes 'out' in vertex shader and 'in' in fragment shader
out vec4 fragColor;  // in webgl2, the builtin gl_FragColor no longer exists, so define an output variable for the fragment color

#define PI 3.14159265
#define TAU (2.0*PI)

#define cx_mul(a, b) vec2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x)

// homemade file include feature, the javascript code will paste the file in here
#include-log-polar-transform.frag

float inverseLerpClamped(float edge0, float edge1, float x)
{
	return clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
}

// Overlay Blending Function
vec3 overlayBlend(vec3 base, vec3 blend)
{
    return mix(2.0 * base * blend, 1.0 - 2.0 * (1.0 - base) * (1.0 - blend), step(0.5, blend));
}

// Luminosity Blending Function
vec3 luminosityBlend(vec3 base, vec3 blend, float amount)
{
    float lumBase = dot(base, vec3(0.3, 0.59, 0.11)); // Calculate luminance of base
    float lumBlend = dot(blend, vec3(0.3, 0.59, 0.11)); // Calculate luminance of blend
    return base + (lumBlend - lumBase) * amount; // Adjust base by difference in luminance
}

// precondition: 0 <= s <= smax
vec3 mapBlackWhite(float s, float smax)
{
	float step = smax / 2.0;
	if (s < step)
	{
		float ratio = s / step;
		return vec3(ratio, ratio, ratio);
	}
	s -= step;
	// if (s < step)
	{
		float ratio = 1.0 - s / step;
		return vec3(ratio, ratio, ratio);
	}
}

// precondition: 0 <= s <= smax
vec3 mapRainbow(float s, float smax)
{
	float step = smax / 6.0;
	if (s < step)
	{
		float ratio = s / step;
		return vec3(1.0, ratio, 0.0);
	}
	s -= step;
	if (s < step)
	{
		float ratio = s / step;
		return vec3(1.0-ratio, 1.0, 0.0);
	}
	s -= step;
	if (s < step)
	{
		float ratio = s / step;
		return vec3(0.0, 1.0, ratio);
	}
	s -= step;
	if (s < step)
	{
		float ratio = s / step;
		return vec3(0.0, 1.0-ratio, 1.0);
	}
	s -= step;
	if (s < step)
	{
		float ratio = s / step;
		return vec3(ratio, 0.0, 1.0);
	}
	s -= step;
	// if (s < step)
	{
		float ratio = s / step;
		return vec3(1.0, 0.0, 1.0-ratio);
	}
}

vec3 getColorForPhase(int oscillatorColors, vec2 oscillatorPhaseVec)
{
	float oscillatorPhaseAngle = atan(oscillatorPhaseVec.y, oscillatorPhaseVec.x) + PI;
	vec3 phaseColor;
	// how to visualize oscillator phase?
	if (oscillatorColors == 0)
		// rainbow color:
		phaseColor = mapRainbow(oscillatorPhaseAngle, TAU);
	else if (oscillatorColors == 1)
		// rainbow color:
		phaseColor = mapRainbow(mod(oscillatorPhaseAngle*2.0, TAU), TAU);
	else if (oscillatorColors == 2 || oscillatorColors == 3)
		// complementary colors
		phaseColor = mapRainbow(oscillatorPhaseAngle, TAU);
	else if (oscillatorColors == 4)
		// blue-yellow sharp version. blue borders grow, then blue areas slowly fade to yellow, then get invaded by new sharp blue borders:
		phaseColor = mix(vec3(0.0, 0.4, 1.0), vec3(1.0, 0.7, 0.0), oscillatorPhaseAngle / TAU);
	else if (oscillatorColors == 5)
		// green-magenta sharp version. green borders grow, then green areas slowly fade to magenta, then get invaded by new sharp green borders:
		phaseColor = mix(vec3(0.0, 1.0, 0.0), vec3(1.0, 0.0, 1.0), oscillatorPhaseAngle / TAU);
		// phaseColor *= (oscillatorPhaseVec.y	+ 1.0) / 2.0;
	else if (oscillatorColors == 6 || oscillatorColors == 7)
		// smooth sine wave back and forth between black and white:
		phaseColor = mix(vec3(0.0, 0.0, 0.0), vec3(1.0, 1.0, 1.0), (oscillatorPhaseVec.x + 1.0) / 2.0);
		// vec3 phaseColor = mix(vec3(0.0, 0.0, 0.0), vec3(1.0, 1.0, 1.0), (oscillatorPhaseVec.x + oscillatorPhaseVec.y + 0.5));
	else if (oscillatorColors == 8)
	{
		// black and white, sharp
		if (oscillatorPhaseVec.x < 0.0)
			phaseColor = vec3(0.0, 0.0, 0.0);
		else
			phaseColor = vec3(1.0, 1.0, 1.0);
	}
	// sawtooth wave between black and white
	// phaseColor = mapBlackWhite(oscillatorPhaseAngle, TAU);
	return phaseColor;
}

vec4 maybeLogPolarTransform(sampler2D tex, int textureIndex, vec2 uv)
{
	// what's the difference between uv and vTexCoord here?
	// vTexCoord is guaranteed to be the raw unmodified value
	// uv is sometimes displaced according to displacement map
	if (logPolarTransform[textureIndex])
	{
		ivec2 itexWidthHeight = textureSize(tex, 0);
		vec2 texWidthHeight = vec2(itexWidthHeight);
		return texture(tex, doLogPolarTransform(texWidthHeight, vTexCoord));
		// return texture(tex, doLogPolarInverseTransform(texWidthHeight, doLogPolarTransform(texWidthHeight, vTexCoord))); // just to check my math
	}
	else
		return texture(tex, uv);
}

void drawFrequency()
{
	// hmm, tricky to visualize frequency in a pedagogical way
	// i settled on a green-magenta oscillation AND a bright-dark gradient combined
	// also it looks best when frameIndex is not huge, so the caller should reset frameIndex to 0 periodically
	vec2 uv = vTexCoord;
	// vec4 oscillatorTexel = texture(uOscillatorTexRead[0], uv);
	vec4 oscillatorTexel = maybeLogPolarTransform(uOscillatorTexRead[0], 0, uv);
	float frequency = oscillatorTexel.b;
	float frequency01 = inverseLerpClamped(frequencyRange[0], frequencyRange[1], frequency);
	float phaseAngle = TAU * frequency * deltaTime * float(frameIndex + 120); // adding 120 is arbitrary, just moves us slightly forward in time, where oscillators are less in phase
	vec2 oscillatorPhaseVec = vec2(cos(phaseAngle), sin(phaseAngle));
	// float oscillatorPhaseVecX01 = (oscillatorPhaseVec.x + 1.0) / 2.0;
	vec3 frequencyColor = vec3(frequency01, frequency01, frequency01);
	// vec3 phaseColor = getColorForPhase(oscillatorColors, oscillatorPhaseVec);
	vec3 phaseColor = getColorForPhase(2, oscillatorPhaseVec);
	// fragColor = vec4(phaseColor * frequency01, 1.0);
	// fragColor = vec4(mix(phaseColor, frequencyColor, 0.5), 1.0);
	fragColor = vec4(frequencyColor + 0.5 * (phaseColor - vec3(0.5, 0.5, 0.5)), 1.0);
}

// a square grid of circles that illustrate how big the coupling kernel is at each point in the image
// outside the circles, draw black and return true
// inside the circles, return false so we instead move on to drawMainEffects()
bool drawKernelSize()
{
	// #TODO: the circles look wobbly when using logPolar transform - I should probably do things in a different order here

	ivec2 itexWidthHeight = ivec2(1, 1);
	if (debugDrawKernelSizeForField == 0)
		itexWidthHeight = textureSize(uOscillatorTexRead[0], 0);
	else if (debugDrawKernelSizeForField == 1)
		itexWidthHeight = textureSize(uOscillatorTexRead[1], 0);

	vec2 texWidthHeight = vec2(itexWidthHeight);
	vec2 smoothTexelCoord = vTexCoord * texWidthHeight;
	// round to the nearest point of an imagined grid. grid cell width is kernelMaxDistance
	smoothTexelCoord /= (kernelMaxDistance * 2.0);
	vec2 blockyTexelCoord = round(smoothTexelCoord) * (kernelMaxDistance * 2.0);
	vec2 uv = blockyTexelCoord / texWidthHeight;

	vec4 oscillatorTexelThere = vec4(0., 0., 0., 0.);
	if (debugDrawKernelSizeForField == 0)
		oscillatorTexelThere = maybeLogPolarTransform(uOscillatorTexRead[0], debugDrawKernelSizeForField, uv);
	else if (debugDrawKernelSizeForField == 1)
		oscillatorTexelThere = maybeLogPolarTransform(uOscillatorTexRead[1], debugDrawKernelSizeForField, uv);

	float adjustedMaxDistance = kernelMaxDistance * oscillatorTexelThere.a;
/*
	int loopRadius = int(ceil(kernelMaxDistance));
	float smallestShrinkFactorFound = 1.0;
	for (int dy = -loopRadius; dy <= loopRadius; dy++)
	{
		for (int dx = -loopRadius; dx <= loopRadius; dx++)
		{
			ivec2 neighborPos = ivec2(mod(blockyTexelCoord.xy + vec2(dx, dy) + texWidthHeight, texWidthHeight));
			vec4 neighbor = texelFetch(uOscillatorTexRead[0], neighborPos, 0);
			smallestShrinkFactorFound = min(smallestShrinkFactorFound, neighbor.a);
		}
	}
*/
	if (length((uv - vTexCoord) * texWidthHeight) < adjustedMaxDistance)
	{
		// fragColor = vec4(1.0, 1.0, 1.0, 1.0);
		return false;
	}
	else
	{
		fragColor = vec4(0.0, 0.0, 0.0, 1.0);
		return true;
	}
}

void drawMainEffects()
{
	vec2 uv = vTexCoord;
	int domainIndex = 0;
	float domainSync = 1.0;
	float domainChaosValue = 0.0;
	if (useDomains && numDomains > 0) {
		vec4 domainMask = texture(uDomainMaskTex, vTexCoord);
		float maxDomains = max(1.0, float(numDomains - 1));
		float domainIndexFloat = domainMask.r * maxDomains;
		domainIndex = clamp(int(floor(domainIndexFloat + 0.5)), 0, numDomains - 1);
		domainSync = domainOrderParameter[domainIndex];
		domainChaosValue = domainChaosParameter[domainIndex];
	}
	float domainChaosClamped = clamp(domainChaosValue, 0.0, 1.0);
	float domainSyncClamped = clamp(domainSync, 0.0, 1.0);
	float csd = clamp(corticalDepressionIntensity, 0.0, 1.0);
	float combinedChaos = clamp(domainChaosClamped + csd * 0.6, 0.0, 1.0);

	// Sample from displacement map (used as a mask)
	// scaled and tiled so it fills the whole canvas
	vec4 dispMapColor = texture(uDisplacementMap, uv * driftPatternScale);
	vec4 blendPatternColor = texture(uBlendingPattern, uv * blendPatternScale);

	// Displace UV coordinates using displacement map for both layers
	uv.x += dispMapColor.r * driftMaxDisplacementHoriz;
	uv.y += dispMapColor.g * driftMaxDisplacementVert;

	vec4 oscillatorTexel0 = maybeLogPolarTransform(uOscillatorTexRead[0], 0, uv);
	vec2 oscillatorPhaseVec = oscillatorTexel0.rg;
	float oscillatorPhaseAngle = atan(oscillatorPhaseVec.y, oscillatorPhaseVec.x);
	vec2 warpedUv = uv + oscillatorPhaseVec * (0.008 * combinedChaos + 0.006 * csd);
	vec3 phaseColor = getColorForPhase(oscillatorColors, oscillatorPhaseVec);
	if (uNumOscillatorTextures > 1)
	{
		phaseColor *= (1.0 - crossFieldBlendingBalance);

		vec4 oscillatorTexel1 = maybeLogPolarTransform(uOscillatorTexRead[1], 1, uv);

		// option 1: store phase as angle between 0 and 2pi
		// float oscillatorPhaseAngle = oscillatorTexel1.g;
		// vec2 oscillatorPhaseVec = vec2(cos(oscillatorPhaseAngle), sin(oscillatorPhaseAngle));
		// option 2: store phase as complex number. leads to better gl.LINEAR interpolation as we scale the texture
		if (oscillatorColors == 2)
		{
			vec2 one = vec2(oscillatorTexel1.g, 0.0); // this one oscillates between two complementary colors in the color wheel...
			// the other one rotates it around the color wheel between 0 and 360 degrees
			oscillatorPhaseVec = cx_mul(one, oscillatorPhaseVec); // rotate
		}
		else if (oscillatorColors == 3)
		{
			vec2 one = vec2(oscillatorTexel1.g, 0.0); // this one oscillates between two complementary colors in the color wheel...
			// the other one rotates it around the color wheel between 45 and 135 degrees
			float rotateAngle = PI*0.5 * (0.5 + (oscillatorPhaseVec.x + 1.0) / 2.0);
			vec2 two = vec2(cos(rotateAngle), sin(rotateAngle));
			oscillatorPhaseVec = cx_mul(one, two);
		}
		else if (oscillatorColors == 7 || oscillatorColors == 8)
		{
			float balance = 2.0*crossFieldBlendingBalance - 1.0; // scale from [0;1] to [-1;1]
			float shiftFor1st = max(0.0, balance); // cut off half, so [0;1] remains
			float shiftFor2nd = max(0.0, -balance); // cut off the other half, and invert, so this one is also [0;1] now, but only one is >0 at the same time
			oscillatorPhaseVec.x = (oscillatorPhaseVec.x + shiftFor1st) * (oscillatorTexel1.r + shiftFor2nd);
		}
		else
			oscillatorPhaseVec *= oscillatorTexel1.rg;

		int secondFieldColors = oscillatorColors;
		if (oscillatorColors == 4) // if the first field is green-magenta,
			secondFieldColors = 5; // the second field is blue-yellow

		phaseColor += getColorForPhase(secondFieldColors, oscillatorTexel1.rg) * crossFieldBlendingBalance;
	}
	float oscillatorPhaseLength = 1.0; // length(oscillatorPhaseVec); // usually 1.0 but not if we did interference between several waves

	// uv += oscillatorPhaseVec * driftMaxDisplacementHoriz; // no, this gives a "frosted glass" effect
	// #TODO: how to do displacement without a frosted glass effect? i guess the displacement must be much smaller than the pattern size

	// Sample from main image (displaced)
	vec2 clampedWarpedUv = clamp(warpedUv, 0.0, 1.0);
	vec4 mainColor = texture(uMainTex, clampedWarpedUv);

	// Sample from custom texture (image or video) (displaced)
	vec3 deepDreamColorOriginal = texture(uDeepDreamTex, clampedWarpedUv).rgb;

	// Apply Overlay blending between main image and custom texture
	// vec3 blendedColor = overlayBlend(mainColor.rgb, deepDreamColor.rgb);

	// Use blendPattern as a mask for transparency (red channel)
	float maskAlpha = blendPatternColor.r;
	// float maskAlpha = 1.0;

	// Apply luminosity blending between main image and custom texture
	vec3 luminosityBlendedColor = luminosityBlend(mainColor.rgb, deepDreamColorOriginal, 1.0);

	// How colorful should the DeepDream be?
	vec3 deepDreamColorMid = mix(luminosityBlendedColor, deepDreamColorOriginal, deepLuminosityBlendOpacity);

	// Apply domain-aware adjustments before compositing
	vec3 domainTint = mix(vec3(1.0, 1.0, 1.0), vec3(1.12, 0.92, 0.85), combinedChaos);
	float saturationBoost = mix(0.75, 1.4, domainSyncClamped) * mix(1.0, 0.7, csd);
	vec3 deepDreamLuma = vec3(dot(deepDreamColorMid, vec3(0.299, 0.587, 0.114)));
	deepDreamColorMid = mix(deepDreamLuma, deepDreamColorMid, clamp(saturationBoost, 0.0, 1.0));
	deepDreamColorMid = mix(deepDreamColorMid, deepDreamColorMid * domainTint, 0.25 * combinedChaos + 0.2 * csd);
	deepDreamColorMid = mix(deepDreamColorMid, deepDreamLuma, 0.3 * csd);

	// Mix based on custom texture opacity and mask alpha
	vec4 deepDreamColorMasked = vec4(mix(mainColor.rgb, deepDreamColorMid, deepDreamOpacity * maskAlpha), 1.0);

	if (oscillatorColors == 2 || oscillatorColors == 3 || oscillatorColors == 7 || oscillatorColors == 8)
		phaseColor = getColorForPhase(oscillatorColors, oscillatorPhaseVec);

	phaseColor = mix(phaseColor, phaseColor * domainTint, 0.45 * combinedChaos);
	phaseColor = mix(phaseColor, vec3(0.6, 0.65, 0.7), 0.3 * csd);
	float phaseGlow = sin(oscillatorPhaseAngle + float(frameIndex) * 0.02);
	phaseColor += 0.1 * combinedChaos * phaseGlow * phaseColor;
	phaseColor = clamp(phaseColor, 0.0, 1.0);

	if (oscillatorColors == 6 || oscillatorColors == 7 || oscillatorColors == 8) // black & white
	{
		// if mixed normally, these colors make the image feel so damn gray. let's try another mixing method
		// float sineWave = phaseColor.r * 2.0 - 1.0; // now it goes from -1 to 1
		// fragColor = vec4(deepDreamColorMasked.rgb + sineWave * oscillatorOpacity * oscillatorPhaseLength, 1.0);
		// fragColor = vec4(mix(deepDreamColorMasked.rgb, overlayBlend(deepDreamColorMasked.rgb, phaseColor), oscillatorOpacity * oscillatorPhaseLength), 1.0);
		fragColor = vec4(luminosityBlend(deepDreamColorMasked.rgb, phaseColor, oscillatorOpacity * oscillatorPhaseLength), 1.0);
	}
	else
	{
		fragColor = mix(deepDreamColorMasked, vec4(phaseColor, 1.0), oscillatorOpacity * oscillatorPhaseLength);
	}
	fragColor.rgb = mix(fragColor.rgb, fragColor.rgb * domainTint, 0.2 * combinedChaos);
	fragColor.rgb = mix(fragColor.rgb, vec3(dot(fragColor.rgb, vec3(0.299, 0.587, 0.114))), 0.35 * csd);
	fragColor.rgb *= mix(1.0, 0.75, csd);
	fragColor.rgb = clamp(fragColor.rgb, 0.0, 1.0);

	// fragColor = blendPatternColor;

	// render the coupled oscillators, just for debugging
	// fragColor = vec4(phaseColor, 1.0);
	// vec3 frequencyColor = vec3(oscillatorTexel.b/7.0, oscillatorTexel.b/7.0, oscillatorTexel.b/7.0);
	// fragColor = oscillatorTexel0;
	// fragColor = vec4(0.0, 0.0, oscillatorTexel0.b/1.1, 1.0);
	// fragColor = mix(deepDreamColorMasked, vec4(frequencyColor, 1.0), deepLuminosityBlendOpacity);
	// fragColor = vec4(frequencyColor * deepLuminosityBlendOpacity, 1.0);
	// fragColor = mix(deepDreamColorMasked, oscillatorTexel, deepLuminosityBlendOpacity);
	// fragColor = mix(deepDreamColorMasked, vec4(oscillatorTexel.b, oscillatorTexel.b, oscillatorTexel.b, 1.0), deepLuminosityBlendOpacity);
	// fragColor = vec4(oscillatorTexel.b/3.3, oscillatorPhaseAngle/TAU, oscillatorTexel.b, oscillatorTexel.a);
	// fragColor = vec4(oscillatorTexel.b/20.0, 0.0, 0.0, oscillatorTexel.a);
	// if (isnan(oscillatorTexel.b))
	// 	fragColor = vec4(1.0, 0.0, 0.0, 1.0);
	// else if (isinf(oscillatorTexel.b))
	// 	fragColor = vec4(1.0, 0.0, 1.0, 1.0);
	// else if (oscillatorTexel.g == 0.0 && oscillatorTexel.b == 0.0)
	// 	fragColor = vec4(1.0, 1.0, 0.0, 1.0);
	// else if (oscillatorTexel.g <= -1000.0)
	// 	fragColor = vec4(1.0, 0.0, 0.0, 1.0);
	// else if (oscillatorTexel.g >= 1000.0)
	// 	fragColor = vec4(1.0, 1.0, 0.0, 1.0);
	// else if (oscillatorTexel.g >= -1000.0)
	// 	fragColor = vec4(0.0, 1.0, 0.0, 1.0);
	// else if (oscillatorTexel.g <= 1000.0)
	// 	fragColor = vec4(0.0, 1.0, 1.0, 1.0);
	// else
	// 	fragColor = vec4(0.0, 10.0 + oscillatorTexel.g, -oscillatorTexel.g, 1.0);
	// fragColor = vec4(1.0, 0.0, 0.0, 1.0);
}

void main()
{
	if (debugDrawFrequency)
		drawFrequency();
	else if (debugDrawKernelSize)
	{
		if (!drawKernelSize())
			drawMainEffects();
	}
	else
		drawMainEffects();
}

