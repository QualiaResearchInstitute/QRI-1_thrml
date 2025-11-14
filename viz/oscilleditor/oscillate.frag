#version 300 es

// this shader does the Coupled Kuramoto Oscillators simulation.
// we use double buffering - we read from one texture and render to another,
// then next time, the source and target are swapped.
// the textures in question must be in floating-point format, not bytes.
// we use the color channels like so:
// RG channels: oscillator current phase, stored as a complex number. R is the real part and G is the imaginary part. changes all the time as the oscillators oscillate
// B channel: oscillator frequency in Hz. this shader just passes it through unchanged. typically set once at startup, then only changed rarely, see init-oscillators.frag
// A channel: coupling kernel size multiplier. this shader just passes it through unchanged. typically set once at startup, then only changed rarely, see init-oscillators.frag

precision mediump float;

// uniform bool debugDrawFilterKernel;

uniform sampler2D uOscillatorTexRead;
//uniform sampler2D uOscillatorTexWrite; no, that's not how it works. we write to gl_FragColor which will end up in a texture
uniform sampler2D uOscillatorTexOther;
uniform sampler2D uDomainMaskTex; // Domain membership texture (R channel = domain index normalized)
uniform float deltaTime;
uniform float couplingToOtherField;
uniform bool logPolarTransformMe;
uniform bool logPolarTransformOther;
uniform bool useDomains; // Whether to use domain-specific parameters
uniform int numDomains; // Number of domains (max 8 for now)
uniform vec4 domainKernelRingCoupling[8]; // Per-domain coupling parameters
uniform vec4 domainKernelRingDistances[8]; // Per-domain distance parameters
uniform vec4 domainKernelRingWidths[8]; // Per-domain width parameters
uniform float domainFrequencyMin[8]; // Per-domain frequency min
uniform float domainFrequencyMax[8]; // Per-domain frequency max
uniform float corticalDepressionIntensity;
uniform float domainOrderParameter[8]; // Per-domain order parameter (R)
uniform float domainChaosParameter[8]; // Per-domain chaos metric (1-R)

// note that more uniforms are #included from a separate file


in vec2 vTexCoord; // in webgl2, 'varying' becomes 'out' in vertex shader and 'in' in fragment shader
out vec4 fragColor;  // in webgl2, the builtin gl_FragColor no longer exists, so define an output variable for the fragment color

#define PI 3.14159265
#define TAU (2.0*PI)

#define cx_mul(a, b) vec2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x)
#define cross2d(a, b) (a.x*b.y - b.x*a.y)


// homemade file include feature, the javascript code will paste the file in here
#include-kernel-ring-function.frag
#include-log-polar-transform.frag


float sineHill(float x)
{
	// x is the distance from the center
	// returns a value between 0 and 1, inclusive
	if (x >= PI)
		return 0.0;
	return (cos(x) + 1.0) / 2.0;
}

// Domain-aware version of influenceFunction2 (must be defined at top level, not nested)
float domainAwareInfluenceFunction2(float sampleDistanceFromOrigo, float inverseMaxDistance, vec4 coupling, vec4 distances, vec4 widths) {
	sampleDistanceFromOrigo *= inverseMaxDistance;
	vec4 sampleDistancesFromRingPeak = sampleDistanceFromOrigo - distances;
	vec4 totalInfluences = coupling * exp(
		-(sampleDistancesFromRingPeak * sampleDistancesFromRingPeak)
		/
		(2.0 * widths * widths)
	);
	return totalInfluences.x + totalInfluences.y + totalInfluences.z + totalInfluences.w;
}

void drawOscillators()
{
	vec2 uv = vTexCoord;
	vec4 oscillatorPrevVal = texture(uOscillatorTexRead, uv);
	float myNaturalFrequency = oscillatorPrevVal.b;
	// float myPhase = oscillatorPrevVal.r;
	vec2 myPhase = oscillatorPrevVal.rg;
	float myPhaseAngle = atan(myPhase.y, myPhase.x);

	// https://webgl2fundamentals.org/webgl/lessons/webgl2-whats-new.html
	// In WebGL1 if your shader needed to know the size of a texture you had to pass the size in uniform manually. In WebGL2 you can call textureSize
	ivec2 itexWidthHeight = textureSize(uOscillatorTexRead, 0);
	vec2 texWidthHeight = vec2(itexWidthHeight);
	vec2 smoothTexelCoord = vTexCoord * texWidthHeight;
	vec2 blockyTexelCoord = floor(smoothTexelCoord); // to index into the texture, use this instead of gl_FragCoord which indexes into the canvas
	// vec4 test = texelFetch(uOscillatorTexRead, ivec2(gl_FragCoord.xy), 0);
	// https://docs.gl/sl4/texelFetch

	// Domain-aware influence function that accepts parameters
	vec4 domainKernelRingCoupling_local = kernelRingCoupling;
	vec4 domainKernelRingDistances_local = kernelRingDistances;
	vec4 domainKernelRingWidths_local = kernelRingWidths;
	float domainFreqMinLocal = myNaturalFrequency;
	float domainFreqMaxLocal = myNaturalFrequency;
	float domainSync = 1.0;
	float domainChaosValue = 0.0;
	float globalDepression = clamp(corticalDepressionIntensity, 0.0, 1.0);
	
	if (useDomains && numDomains > 0) {
		vec4 domainMask = texture(uDomainMaskTex, uv);
		// Domain index is stored in R channel, normalized to [0,1]
		// Convert back to integer domain index
		float numDomainsFloat = float(numDomains);
		float maxDomains = max(1.0, numDomainsFloat - 1.0);
		float domainIndexFloat = domainMask.r * maxDomains;
		int domainIndex = int(floor(domainIndexFloat + 0.5)); // Round to nearest
		domainIndex = clamp(domainIndex, 0, numDomains - 1);
		
		// Use domain-specific parameters
		domainKernelRingCoupling_local = domainKernelRingCoupling[domainIndex];
		domainKernelRingDistances_local = domainKernelRingDistances[domainIndex];
		domainKernelRingWidths_local = domainKernelRingWidths[domainIndex];
		domainSync = domainOrderParameter[domainIndex];
		domainChaosValue = domainChaosParameter[domainIndex];
		domainFreqMinLocal = domainFrequencyMin[domainIndex];
		domainFreqMaxLocal = domainFrequencyMax[domainIndex];
	}


/*
	// alternate idea: couple with similar colors
	// anti-couple with different colors
	// worked, but results seems quite boring
	// expects 		gl.bindTexture(gl.TEXTURE_2D, g_mainImageTexture);
	for (int dy = -kernelMaxDistance; dy <= kernelMaxDistance; dy++)
	{
		for (int dx = -kernelMaxDistance; dx <= kernelMaxDistance; dx++)
		{
			if (dx != 0 || dy != 0)
			{
				// https://stackoverflow.com/questions/16825412/what-is-the-range-of-gl-fragcoord
				// at the edges, use pacman wraparound aka torus topology
				vec2 neighborPos = mod(gl_FragCoord.xy + vec2(dx, dy) + texWidthHeight, texWidthHeight);
				vec4 neighborOscillator = texelFetch(uOscillatorTexRead, ivec2(neighborPos), 0);
				vec4 neighborPhotoPixel = texelFetch(uDepthOrEdgeTex, ivec2(neighborPos), 0);
			    float phaseDifference = neighborOscillator.g - myPhase;
			    float distance = sqrt(float(dx * dx + dy * dy));
			    vec3 photoColorDiff = neighborPhotoPixel.rgb - myPhotoPixel.rgb;
			    // maximum color diff is 1 in all 3 dimensions
			    // cuberoot(1 + 1 + 1) = 1.4423
			    float difference = length(photoColorDiff) / 1.4423;
				sumCouplingStrength += 10.0 * difference * sineHill(PI * distance / kernelMaxDistance);
			    float similarity = (1.0 - length(photoColorDiff) / 1.4423); // similarity is now between 0 and 1
			    similarity = similarity * similarity * similarity;
			    similarity = 2.0 * similarity - 1.0; // similarity is now between -1 and 1
			    // float couplingStrength = 1.0 * similarity;
			    float couplingStrength = 100.0 * similarity * sineHill(PI * distance / kernelMaxDistance);
			    // float couplingStrength = 1.0 * sineHill(PI * distance / kernelMaxDistance);
			    couplingTerm += couplingStrength * sin(phaseDifference);
	 		    numNeighbors++;
			}
		}
	}
*/

	// #TODO: small world neighbors
	// Calculate coupling term for small world neighbors
	// for (float neighbor of dot.smallWorldNeighbors) {
	//   float phaseDifference = neighbor.phase - myPhase;
	//   couplingTerm += couplingSmallWorld * sin(phaseDifference);
	//   numNeighbors++;
	// }

	// const float deltaTime = 1.0 / 60.0; // Time step for numerical integration. If you change this, change in shader.frag drawFrequency() too!

	float couplingTerm = 0.0;
	// vec2 couplingTerm = vec2(0.0, 0.0);
	float sumCouplingStrength = 0.0;
	// int numNeighbors = 0;

	// apply convolution kernel
	// scale the coupling kernel by texture channel A (which was based on image depth map or edge map in init-oscillators.frag)
	// if you want to change the scaling to something more advanced, change there, not here! keep this a simple multiplication!
	float adjustedMaxDistance = kernelMaxDistance * oscillatorPrevVal.a;
	float inverseMaxDistance = 1.0 / adjustedMaxDistance;
	int loopRadius = int(ceil(adjustedMaxDistance));
	for (int dy = -loopRadius; dy <= loopRadius; dy++)
	{
		for (int dx = -loopRadius; dx <= loopRadius; dx++)
		{
			if (dx != 0 || dy != 0)
			{
				// ivec2 neighborPos = ivec2(blockyTexelCoord.xy + vec2(dx, dy));
				// if (neighborPos.x >= 0 && neighborPos.x < itexWidthHeight.x && neighborPos.y >= 0 && neighborPos.y < itexWidthHeight.y)

				// at the edges, use pacman wraparound aka torus topology
				ivec2 neighborPos = ivec2(mod(blockyTexelCoord.xy + vec2(dx, dy) + texWidthHeight, texWidthHeight));

				vec4 neighbor = texelFetch(uOscillatorTexRead, neighborPos, 0);
			    float distance = sqrt(float(dx * dx + dy * dy));
			    // float distance = length(vec2(dx, dy));
			    // float couplingStrength = influenceFunction(distance, adjustedMaxDistance);
			    // Use domain-aware influence function if domains are enabled
			    float couplingStrength;
			    if (useDomains && numDomains > 0) {
			    	couplingStrength = domainAwareInfluenceFunction2(distance, inverseMaxDistance, 
			    		domainKernelRingCoupling_local, domainKernelRingDistances_local, domainKernelRingWidths_local);
			    } else {
			    	couplingStrength = influenceFunction2(distance, inverseMaxDistance); // faster?
			    }
				float domainCouplingScale = mix(0.75, 1.25, clamp(domainChaosValue, 0.0, 1.0));
				float depressionCouplingScale = 1.0 - 0.75 * globalDepression;
				couplingStrength *= domainCouplingScale * depressionCouplingScale;


				// option 1: store phase as angle between 0 and 2pi
			    // float phaseDifference = neighbor.r - myPhase;


				// option 2: store phase as complex number. leads to better gl.LINEAR interpolation as we scale the texture

				// option 2a: we convert to angle temporarily
				float neighborPhaseAngle = atan(neighbor.g, neighbor.r);
				float phaseDifference = neighborPhaseAngle - myPhaseAngle;
				float sineOfPhaseDifference = sin(phaseDifference);

				// option 2b: doesn't work
			    // float ourDot = dot(neighbor.rg, myPhase);
			    // float sineOfPhaseDifference = sqrt(1.0 - ourDot * ourDot);

				// option 2c: doesn't work
			    // float sineOfPhaseDifference = cross2d(neighbor.rg, myPhase);


				couplingTerm += couplingStrength * sineOfPhaseDifference;
				sumCouplingStrength += abs(couplingStrength);
	 		    // numNeighbors++;
			}
		}
	}
	// how to "normalize" the coupling kernel?
	// couplingTerm /= float(numNeighbors); // no, dividing by number of neighbors feels wrong
	// if we would sample a larger square far outside the rings' perimeter, the average would get lower (because numNeighbors would be larger), but it should stay the same.
	// if we instead do  couplingTerm /= sumCouplingStrength;  we get "correct" behaviour except that everything is always completely normalized,
	// ie only the relative strengths between the coupling sliders matter, not their absolute values. that doesn't feel right either.
	// so multiply by the absolute sum of all 4 coupling sliders, so that they start to matter again
	float absSumKernelRingCoupling;
	if (useDomains && numDomains > 0) {
		absSumKernelRingCoupling = (abs(domainKernelRingCoupling_local.x) + abs(domainKernelRingCoupling_local.y) + 
			abs(domainKernelRingCoupling_local.z) + abs(domainKernelRingCoupling_local.w));
	} else {
		absSumKernelRingCoupling = (abs(kernelRingCoupling.x) + abs(kernelRingCoupling.y) + abs(kernelRingCoupling.z) + abs(kernelRingCoupling.w));
	}
	if (sumCouplingStrength != 0.0)
		couplingTerm = couplingTerm * absSumKernelRingCoupling / sumCouplingStrength;
	couplingTerm *= (1.0 - 0.6 * globalDepression);

	if (useDomains && numDomains > 0) {
		float freqTarget = mix(domainFreqMinLocal, domainFreqMaxLocal, clamp(domainSync, 0.0, 1.0) * 0.5 + 0.25);
		myNaturalFrequency = mix(myNaturalFrequency, freqTarget, 0.2 * clamp(domainChaosValue, 0.0, 1.0) + 0.3 * globalDepression);
	}
	myNaturalFrequency = mix(myNaturalFrequency, myNaturalFrequency * 0.35, globalDepression);

	if (couplingToOtherField != 0.0)
	{
		// should we just take a single sample from the other field?
		// it feels more numerically stable to take a bunch of samples, at least when the fields have different resolutions.
		// calculate how big each texel becomes in the other space
		vec2 otherTexWidthHeight = vec2(textureSize(uOscillatorTexOther, 0));
		float widthOfOurTexelInOtherField = otherTexWidthHeight.x / texWidthHeight.x;
		int loopDiameter = int(round(widthOfOurTexelInOtherField));
		if (loopDiameter <= 1)
		{
			vec4 neighbor;
			if (logPolarTransformMe == logPolarTransformOther)
				neighbor = texture(uOscillatorTexOther, uv);
			else if (logPolarTransformMe && !logPolarTransformOther)
				neighbor = texture(uOscillatorTexOther, doLogPolarInverseTransform(texWidthHeight, uv));
			else if (!logPolarTransformMe && logPolarTransformOther)
				neighbor = texture(uOscillatorTexOther, doLogPolarTransform(texWidthHeight, uv));
			float couplingStrength = couplingToOtherField;// * kernelMaxDistance * kernelMaxDistance; // hack, makes sense intuitively, because the normal kernel also grows larger by the square of kernelMaxDistance
			float neighborPhaseAngle = atan(neighbor.g, neighbor.r);
			float phaseDifference = neighborPhaseAngle - myPhaseAngle;
			float sineOfPhaseDifference = sin(phaseDifference);
			couplingTerm += couplingStrength * sineOfPhaseDifference;
			// sumCouplingStrength += abs(couplingStrength);
			// numNeighbors++;
		}
		else
		{
			// when a low-res field samples from a high-res field, we should probably take more samples
			// #TODO: figure out how to sample with maximal quality when one field is log-polar transformed and the other one isn't
			// i'm not sure if we should transform before or after doing the wraparound, for example
			float couplingTermFromOtherField = 0.0;
			int numNeighborsInOtherField = 0;
			for (int dy = 0; dy < loopDiameter; dy++)
			{
				for (int dx = 0; dx < loopDiameter; dx++)
				{
					// at the edges, use pacman wraparound aka torus topology
					vec2 neighborPos = vec2(mod(uv + (vec2(dx, dy) - widthOfOurTexelInOtherField/2.0) / otherTexWidthHeight + vec2(1.0, 1.0), vec2(1.0, 1.0)));
					vec4 neighbor;
					if (logPolarTransformMe == logPolarTransformOther)
						neighbor = texture(uOscillatorTexOther, neighborPos);
					else if (logPolarTransformMe && !logPolarTransformOther)
						neighbor = texture(uOscillatorTexOther, doLogPolarInverseTransform(texWidthHeight, neighborPos));
					else if (!logPolarTransformMe && logPolarTransformOther)
						neighbor = texture(uOscillatorTexOther, doLogPolarTransform(texWidthHeight, neighborPos));

					float couplingStrength = couplingToOtherField;
					float neighborPhaseAngle = atan(neighbor.g, neighbor.r);
					float phaseDifference = neighborPhaseAngle - myPhaseAngle;
					float sineOfPhaseDifference = sin(phaseDifference);
					// should we simply add to couplingTerm here, just like the above kernel loop?
					// no, that would mean that the couplingStrength to the other fields would become _stronger_ when we have a large resolution difference, which is obviously bogus
					// hacky optimizations like lower texture resolution shouldn't affect the math!
					// so instead, i think it's correct to take a separate average of all samples from the other field, and then treat them as a single sample as in the if (loopDiameter <= 1) case above
					couplingTermFromOtherField += couplingStrength * sineOfPhaseDifference;
					numNeighborsInOtherField++;
				}
			}
			couplingTermFromOtherField /= float(numNeighborsInOtherField);
			float crossFieldScale = mix(0.85, 1.15, clamp(domainChaosValue, 0.0, 1.0)) * (1.0 - 0.5 * globalDepression);
			// float sumCouplingStrengthFromOtherField = couplingToOtherField * kernelMaxDistance * kernelMaxDistance; // hack, makes sense intuitively, because the normal kernel also grows larger by the square of kernelMaxDistance
			couplingTerm += couplingTermFromOtherField * crossFieldScale; // * kernelMaxDistance * kernelMaxDistance; // hack, makes sense intuitively, because the normal kernel also grows larger by the square of kernelMaxDistance
			// sumCouplingStrength += abs(couplingToOtherField);
			// numNeighbors++;
		}
	}


	// Update phase using the Euler method


	// option 1: store phase as angle between 0 and 2pi
	// myPhase += (myNaturalFrequency + couplingTerm) * deltaTime;
	// myPhase = mod(myPhase, TAU);
	// fragColor = vec4(myPhase, sumCouplingStrength / float(numNeighbors), oscillatorPrevVal.b, 0.0);


	// option 2: store phase as complex number. leads to better gl.LINEAR interpolation as we scale the texture
/*
	// update due to natural frequency
	float naturalFrequencyDt = myNaturalFrequency * deltaTime;
	vec2 naturalFrequencyUpdate = vec2(cos(naturalFrequencyDt), sin(naturalFrequencyDt));
	myPhase = cx_mul(myPhase, naturalFrequencyUpdate);

	// update due to coupling to neighbors
	float asdf = couplingTerm * deltaTime;
	vec2 couplingUpdate = vec2(cos(asdf), sin(asdf));
	myPhase = cx_mul(myPhase, couplingUpdate);
*/
	// both updates at once
	float asdf = TAU * (myNaturalFrequency + couplingTerm) * deltaTime;
	vec2 couplingUpdate = vec2(cos(asdf), sin(asdf));
	myPhase = cx_mul(myPhase, couplingUpdate);
	myPhase = normalize(myPhase); // shouldn't be neccessary in pure math land, but numerically it does make a visible difference!

	// #TODO: it should be possible to do the update with less calls to sin & cos, staying with complex numbers
	// below are some broken attempts at this
	// float k = 1.0;
	// vec2 dz = normalize(couplingTerm) - myPhase;
	// vec2 difference = vec2(
	// 	-myNaturalFrequency * myPhase.y + k * dz.x,
	// 	myNaturalFrequency * myPhase.x + k * dz.y
	// );
	// myPhase = normalize(myPhase + difference * deltaTime);
	// myPhase = normalize(myPhase + normalize(couplingTerm) * deltaTime);

	// vec2 localChange = vec2(-myNaturalFrequency * myPhase.y) + (myNaturalFrequency * myPhase.x);

	// myPhase = normalize(myPhase + myNaturalFrequency * couplingTerm * deltaTime);

	// float sineOfPhaseDifference = cross(normalize(couplingTerm), myPhase);
	// vec2 localChange = vec2(-myNaturalFrequency * myPhase.y) + (myNaturalFrequency * myPhase.x);
	// myPhase += localChange * deltaTime;

	fragColor = vec4(myPhase.x, myPhase.y, oscillatorPrevVal.b, oscillatorPrevVal.a);
}

void main()
{
	// debugDrawFilterKernel=true code path outcommented because of a sudden performance regression
	// in commit 5d49d9910e384a1e3112c85cfe2cb3881bf1b88d on 2024-11-23 with description
	// "Show the effects of more GUI actions immediately. Make the play&pause buttons have effect on the video loops too. Don't resize the oscillator textures if you are paused, but still preview the coupling kernel correctly."
	// for some bizzare reason, this code path became slow at large kernel sizes like kernelMaxDistance ~100
	// and if i outcommented the whole drawOscillators code path it got fast again
	// i can only assume we hit some edge case in the glsl compiler so it failed to optimize something that it previously optimized
	// as an annoying workaround, i moved this code path out to the separate shader draw-kernel.frag
	// annoying because now we need to share code between shaders
	// if (debugDrawFilterKernel)
	// 	drawFilterKernel();
	// else
		drawOscillators();
}
