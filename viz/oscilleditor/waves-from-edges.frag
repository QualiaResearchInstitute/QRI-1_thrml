#version 300 es

// this shader does https://qri1.gitlab.io/tactile-visualizer/hypnagogic_quasicrystals.html?Layers=7&Speed=0.2
// but using a bunch of distance field textures

precision mediump float;

uniform float timeSec;
uniform sampler2D uMainTex;         // Main image texture
uniform int uNumDistanceFieldTextures;
uniform sampler2D uDistanceFieldTex[8];
uniform float uLineLength[8];
uniform float uSpeed[8];
uniform float uWavelength[8];
uniform float uFrequencyFalloff[8];
uniform float uAmplitude[8];
uniform float uAmplitudeFalloff[8];
uniform float uBlendOrAlternate[8]; // 1 means to blend both fields equally, -1 means to alternate between them sharply like a square wave
uniform float uAlternationWavelength[8]; // has no visible effect is uBlendOrAlternate is 1
uniform float uShiftAlongStrength[8];
uniform int uDebugDrawDfIndex;
uniform float uOpacity;
uniform float uOpacityTweak;
uniform float uCenterOverlapBefore;
uniform float uCenterOverlapAfter;
uniform float uCurvature;
uniform float uExperimental;


in vec2 vTexCoord; // in webgl2, 'varying' becomes 'out' in vertex shader and 'in' in fragment shader
out vec4 fragColor;  // in webgl2, the builtin gl_FragColor no longer exists, so define an output variable for the fragment color

#define PI 3.14159265
#define TAU (2.0*PI)

float inverseLerp(float from, float to, float value)
{
	return (value - from) / (to - from);
}

float inverseLerpClamped(float from, float to, float value)
{
	return clamp(inverseLerp(from, to, value), 0., 1.);
}

// remaps from the range [min1, max1] to [min2, max2] without clamping
float reMap(float value, float min1, float max1, float min2, float max2)
{
	return mix(min2, max2, inverseLerp(min1, max1, value));
}

float logBaseB(float value, float base)
{
	return log(value) / log(base);
}

// symmetric hill / bump around 0
// the width is approximate
// gaussian(0) is the peak of the hill, returns 1
// gaussian(>>width) returns 0
float gaussian(float x, float width)
{
	return exp(-pow(x*2./width, 2.));
}

// https://iquilezles.org/articles/functions/
// similar to gaussian, but the width is exact
// cubicPulse(2.99, 3) returns >0
// cubicPulse(3, 3) returns 0
float cubicPulse(float x, float width)
{
	x = abs(x);
	if (x > width)
		return 0.0;
	x /= width;
	return 1.0 - x*x*(3.0-2.0*x);
}

// This function can also be a replacement for a gaussian sometimes
// if you can do with infinite support,
// ie, this function never reaches exactly zero no matter how far you go in the real axis
// k is not a width, it goes the opposite way, ie small k means wide hill and big k means narrow hill
float rationalBump(float x, float k)
{
	return 1.0/(1.0+k*x*x);
}

// https://iquilezles.org/articles/functions/
// Remapping the unit interval into the unit interval
// by expanding the sides and compressing the center,
// and keeping 1/2 mapped to 1/2,
// that can be done with the gain() function.
// k=1 is the identity curve
// k<1 produces the classic gain() shape
// k>1 produces "s" shaped curces
// The curves are symmetric (and inverse) for k=a and k=1/a.
float gain( float x, float k ) 
{
    float a = 0.5*pow(2.0*((x<0.5)?x:1.0-x), k);
    return (x<0.5)?a:1.0-a;
}

// https://iquilezles.org/articles/smin/
// quadratic polynomial
float smin( float a, float b, float k )
{
    k *= 4.0;
    float h = max( k-abs(a-b), 0.0 )/k;
    return min(a,b) - h*h*k*(1.0/4.0);
}

// like mix() but uses the geometric mean instead
float mixGeometric(float from, float to, float value)
{
	if (from == 0. || to == 0.) // special case to avoid log(0) which returns -inf or NaN or something
		return 0.;
	return exp(mix(log(from), log(to), value));
}

// like mix() but uses the harmonic mean instead
float mixHarmonic(float from, float to, float value)
{
	if (from == 0. || to == 0.) // special case to avoid 1/0 which returns -inf or NaN or something
		return 0.;
	return 1. / mix(1. / from, 1. / to, value);
	// return 1. / ((1. - value) / from + value / to);
}

// p = period
// a = amplitude
float triangleWave(float x, float p, float a)
{
	return abs(mod(x-p/4., p) - p/2.) * 4.*a/p - a;
}

// p = period
// return range -1 to 1
float sawtoothWave(float x, float p)
{
	return 2. * (x/p - floor(0.5 + x / p));
}

vec2 sawtoothWave2(float x, float p)
{
	float waveIndex = floor(0.5 + x/p);
	float fractionalPart = 2. * (x/p - waveIndex);
	return vec2(fractionalPart, abs(waveIndex));
}

void addVanishingPoint(int i, inout float waveTotal, inout float envelopeTotal, vec2 myFragCoord, float xDistanceToVanishingPointRaw, float opacity)
{
	float uLinearScaling = 0.001 * uWavelength[i];
	//float uLinearScaling = 0.0001 * uWavelength[i] * uAlternationWavelength[i];
	// float uOpacityTweak = uExperimental;
	// const float uCenterOverlapBefore = 35.;
	// const float uCenterOverlapAfter = 26.;
	// const float uCurvature = 0.77;

	float yDistanceToCenterLine = myFragCoord.y;
	yDistanceToCenterLine += uCenterOverlapAfter;

	float xDistanceToVanishingPoint = xDistanceToVanishingPointRaw;// + yDistanceToCenterLine * uStrafeTranslation3D;

	// with uCurvature the wavelength becomes so damn long or short
	// so compensate so the wavelength stays the same in one place on the screen
	float referenceYDistanceBefore = 100.;
	float referenceYDistanceTransformed = referenceYDistanceBefore;

	// the exponent should go from 1/2 to 2
	float curvatureExponent;
	if (uCurvature < 0.)
		curvatureExponent = 1./(1.-uCurvature);
	else
		curvatureExponent = 1.+uCurvature;
	yDistanceToCenterLine = pow(yDistanceToCenterLine, curvatureExponent);
	referenceYDistanceTransformed = pow(referenceYDistanceTransformed, curvatureExponent);

	float oldMax = 180.;
	float newMax = pow(oldMax, curvatureExponent);
	yDistanceToCenterLine += (uCenterOverlapBefore * newMax / oldMax);

	float sawtoothWavelength = yDistanceToCenterLine * uLinearScaling;
	sawtoothWavelength *= (referenceYDistanceBefore / referenceYDistanceTransformed); // #TODO: this scaling factor is invariant across all pixels, so it could be computed on the javascript side and passed in as a uniform, or at least computed outside the addVanishingPoint loop. not sure if the compiler is smart enough to hoist it automatically? although the code is more maintainable this way...

	// what happens if the lines are evenly distributed around the unit circle instead? will break the perspective grid though...
	// float angleToVanishingPoint = atan(xDistanceToVanishingPoint, myFragCoord.y);
	// float angularCompensationFactor = mix(1., tan(angleToVanishingPoint) / angleToVanishingPoint, uAngularness);
	// sawtoothWavelength = sawtoothWavelength * angularCompensationFactor;

	vec2 wave = sawtoothWave2(xDistanceToVanishingPoint, sawtoothWavelength);
	// #TODO: as you adjust uLinearScaling to increase the width of each wave, consider fading in new sub-waves in between? like a shepherd tone

	// #TODO: as you adjust uLinearScaling, it would look better if the line start points didn't move, ie keep the same "apparent" uCenterOverlap
	// uZoom has that desired effect, but it also has other undesired effects: it changes the apperent spacing between vanishing points, and the opacity
	// so maybe i can implement uLinearScaling in terms of uZoom and an "inverse" effect on spacing and opacity?

	// #TODO: tweak the opacity! currently i have three things it could depend on:
	// * the index of the wave, where 0 is straight vertical, 1 is its neighbor with slight tilt, 2 is even more tilted and so on
	// * the horizontal distance, where one distanceAtFullOpacity away is one half-life (opacity gets halved for each such step)
	// * the angle of the wave, measured as "rise over run", which gives a different result than the wave index because wave period can vary
	// think about how to combine them

	float distanceAtFullOpacity = 400.;
	float distanceBeyondThat = max(0., abs(xDistanceToVanishingPoint) - distanceAtFullOpacity);
	float slopeRiseOverRun = abs(myFragCoord.y / xDistanceToVanishingPointRaw);
	opacity *= min(1., pow(slopeRiseOverRun / uOpacityTweak, 2.));
	opacity *= pow(2., - uOpacityTweak * distanceBeyondThat / distanceAtFullOpacity);

	waveTotal += opacity * wave.x;
	envelopeTotal += opacity;
	//envelopeTotal += 1.;
}

void addSingleWave(int i, inout float waveTotal, inout vec4 mainColor, float wavelength, float scaledAmplitude, float scaledDistance)
{
	float localTime = (sin(timeSec) * 0.5 + 0.5) * 6.;
	scaledDistance /= wavelength;
	float wave = sawtoothWave(scaledDistance - localTime * uSpeed[i], 1.) * scaledAmplitude;
	waveTotal += wave;
	// if (uDebugDrawDfIndex == i)
	// {
	// 	vec3 debugColor = vec3(0.0, wave * 0.5, 0.0);
	// 	mainColor = vec4(mainColor.rgb + debugColor, 1.0);
	// }
}

// "perp" is the signed distance to the infinite line
// "along" is the distance along the infinite line, that is, at a 90 degree angle to "perp". this is 0 at one endpoint of the line, and is uLineLength[] at the other endpoint
void addTwoCrossingWaves(int i, float distanceRatio, float distPerpWithPolarity, float distAlong, float distPerpPreserving0s, float distAlongForFalloff, inout float waveTotal, inout float envelopeTotal, inout vec4 mainColor)
{
	// bends the pattern, like spruce trees in a storm
	// #TODO: change the wavelength to compensate, so the wave spacing on screen stays more constant?
	float shiftAlong = abs(distPerpPreserving0s) * uShiftAlongStrength[i] * sin(timeSec + PI / 2.);
	distAlong += shiftAlong;

	// fundamentally, we have two sets of more-or-less parallel lines, intersecting at more-or-less 45 degrees
	// these sets _exist_ at every point in space
	// but we don't have to _render_ them both at every point in space
	// we can alternate between them, more or less smoothly or sharply
	float blendPeriodicity = uAlternationWavelength[i];
	float blendOrAlternate = uBlendOrAlternate[i];
	float alternatingAmplitude = sin(distAlong * PI / blendPeriodicity);
	// float alternatingAmplitude = mix(1., sin(distAlong * PI / blendPeriodicity), 1.-distanceRatio) * mix(1., sin(distPerpWithPolarity * PI / blendPeriodicity), distanceRatio) * 0.5;
	float alternatingAmplitude0;
	float alternatingAmplitude1;
	if (blendOrAlternate < 0.)
	{
		// amplify the sine wave so it goes beyond its normal range, then clip it to [0;1]
		// this makes it look more and more like a square wave the more we amplify it
		float sineClipping = 1. + blendOrAlternate;
		alternatingAmplitude = alternatingAmplitude / sineClipping + 0.5;
		alternatingAmplitude0 = clamp(alternatingAmplitude, 0., 1.);
		alternatingAmplitude1 = 1. - alternatingAmplitude0;
	}
	else
	{
		// reduce the range of the sine wave, so it goes between for example [0.5; 1] or [0.8; 1]
		alternatingAmplitude = alternatingAmplitude + 0.5;
		alternatingAmplitude0 = mix(blendOrAlternate, 1.0, alternatingAmplitude);
		alternatingAmplitude1 = mix(1.0, blendOrAlternate, alternatingAmplitude);
	}
/*
	// beyond the line endpoints, we only show one of the patterns, not both
	// but we don't want an abrupt change, we want to fade out gradually over a short distance
	const float fadeOutHalfDistance = 20.;
	if (distAlongForFalloff / uLineLength[i] < 0.5)
	{
		float distanceToEndpoint = distAlongForFalloff;
		float t = inverseLerpClamped(-fadeOutHalfDistance, fadeOutHalfDistance, distanceToEndpoint);
		alternatingAmplitude0 = mix(1., alternatingAmplitude0, t);
		alternatingAmplitude1 = mix(0., alternatingAmplitude1, t);
	}
	else
	{
		float distanceToEndpoint = uLineLength[i] - distAlongForFalloff;
		float t = inverseLerpClamped(-fadeOutHalfDistance, fadeOutHalfDistance, distanceToEndpoint);
		alternatingAmplitude0 = mix(0., alternatingAmplitude0, t);
		alternatingAmplitude1 = mix(1., alternatingAmplitude1, t);
	}
*/

	// the pattern amplitude should fall off by distance, but how? and what distance?
	float distanceToSegmentAlong = 0.;
	if (distAlongForFalloff < 0.)
		distanceToSegmentAlong = -distAlongForFalloff;
	else if (distAlongForFalloff > uLineLength[i])
		distanceToSegmentAlong = distAlongForFalloff - uLineLength[i];

	// float longestDistance = abs(distPerpPreserving0s);
	float longestDistance = max(abs(distPerpPreserving0s), distanceToSegmentAlong); // max() forms a rectangular distance field
	// float scaledAmplitude = rationalBump(longestDistance, uAmplitudeFalloff[i] / 640.) * uAmplitude[i];
	float scaledAmplitude = gaussian(longestDistance, 640. * uAmplitudeFalloff[i]) * uAmplitude[i];
	// float scaledAmplitude = uAmplitude[i] / ((1. - uAmplitudeFalloff[i]) * longestDistance + 1.);
	// float scaledAmplitudePerp = 1. / ((1. - uAmplitudeFalloff[i]) * abs(distPerpPreserving0s) + 1.);
	// float scaledAmplitudeAlong = 1. / ((1. - uAmplitudeFalloff[i]) * distanceToSegmentAlong + 1.);
	// float scaledAmplitude = uAmplitude[i] * min(scaledAmplitudePerp, scaledAmplitudeAlong);
	// float scaledAmplitude = uAmplitude[i] * scaledAmplitudePerp * scaledAmplitudeAlong;

	alternatingAmplitude0 *= scaledAmplitude;
	alternatingAmplitude1 *= scaledAmplitude;

	//float arbitraryScale = 40.;// * uExperimental; // i don't love such arbitrary constants
	//distPerpWithPolarity *= arbitraryScale;

	float xDistanceToVanishingPointLeft = -mod(distAlong, uAlternationWavelength[i]);
	float xDistanceToVanishingPointRight = uAlternationWavelength[i] + xDistanceToVanishingPointLeft;
	addVanishingPoint(i, waveTotal, envelopeTotal, vec2(distAlong, abs(distPerpWithPolarity)), xDistanceToVanishingPointLeft, scaledAmplitude);
	addVanishingPoint(i, waveTotal, envelopeTotal, vec2(distAlong, abs(distPerpWithPolarity)), xDistanceToVanishingPointRight, scaledAmplitude);
	xDistanceToVanishingPointLeft = xDistanceToVanishingPointLeft - uAlternationWavelength[i];
	addVanishingPoint(i, waveTotal, envelopeTotal, vec2(distAlong, abs(distPerpWithPolarity)), xDistanceToVanishingPointLeft, scaledAmplitude);
	xDistanceToVanishingPointRight = xDistanceToVanishingPointRight + uAlternationWavelength[i];
	addVanishingPoint(i, waveTotal, envelopeTotal, vec2(distAlong, abs(distPerpWithPolarity)), xDistanceToVanishingPointRight, scaledAmplitude);

	// addSingleWave(i, waveTotal, mainColor, uWavelength[i], alternatingAmplitude0, distAlong);
	// addSingleWave(i, waveTotal, mainColor, uWavelength[i], alternatingAmplitude0, distPerpWithPolarity);
	// addSingleWave(i, waveTotal, mainColor, uWavelength[i], alternatingAmplitude0, distPerpWithPolarity + distAlong);
	// addSingleWave(i, waveTotal, mainColor, uWavelength[i], alternatingAmplitude1, distPerpWithPolarity - distAlong);
	// float tRotate = uShiftAlongStrength[i] * sin(timeSec) * 0.5 + 1.;
	// addSingleWave(i, waveTotal, mainColor, uWavelength[i], alternatingAmplitude0, distPerpWithPolarity + distAlong * tRotate);
	// addSingleWave(i, waveTotal, mainColor, uWavelength[i] * 0.67, alternatingAmplitude1, mix(distPerpWithPolarity, -distAlong, tRotate));
	// envelopeTotal += scaledAmplitude;
	//envelopeTotal += 1.;
}

float scaleDistancePerp(float signedDistance)
{
	// float frequencyFalloff = uFrequencyFalloff[i];
	float absDistance = abs(signedDistance);
	// float scaledDistance = logBaseB(absDistance+1., 1./frequencyFalloff); // no, this one always has the same "shape", the falloff just affects the wavelength
	// float scaledDistance = pow(absDistance, frequencyFalloff); // no, this affects the wavelength much more than it affects the "shape"
	// float scaledDistance = mix(10.*pow(absDistance, frequencyFalloff), absDistance, frequencyFalloff); // no, the mixing doesn't feel "linear" enough
	// float scaledDistance = mixGeometric(10.*pow(absDistance, frequencyFalloff), absDistance, frequencyFalloff); // no, the mixing doesn't feel "linear" enough
	// float scaledDistance = mixHarmonic(10.*pow(absDistance, frequencyFalloff), absDistance, frequencyFalloff); // no, the mixing doesn't feel "linear" enough
	// float scaledDistance = mix(10.*log(absDistance+1.), absDistance, frequencyFalloff); // no, the mixing doesn't feel "linear" enough
	float offset = 10.; // to avoid negative infinity. must be >0 but 0.1 gives a lot of aliasing / moire due to really thin lines
	// float scaledDistance = mixGeometric(arbitraryScale * (log(absDistance+offset) - log(offset)), absDistance, frequencyFalloff); // good. subtract log(offset) to avoid negative numbers
	//float scaledDistance = log(absDistance+offset) - log(offset); // good. subtract log(offset) to avoid negative numbers
	float scaledDistance = log(absDistance + offset); // i don't need to avoid negative numbers
	//float arbitraryScale = 40.;// * uExperimental; // i don't love such arbitrary constants
	//scaledDistance *= arbitraryScale;
	return scaledDistance;
}

// almost the same as abs(distance), but with a deeper, sharper valley that goes below 0, instead of the usual turning point at 0
// "a" = 0 gives us exactly abs(distance)
// increasing "a" makes the valley both deeper and wider
// "b" = 0 gives us the classic 1/x shape with infinite valley depth
// increasing "b" makes the valley finite. the depth changes, but not the width. really high "b" approaches abs(distance)
// then i added the sign * multiplication, which means it's no longer a left-right symmetric valley, one half is now upside down, with a sudden step function in between
float scaleDistanceAlong(float distance, float a, float b)
{
	// return distance;
	return sign(distance) * (abs(distance) - a / (abs(distance) + b));
}

float combinePerpAsCuspsWithOppositePolarities(float main, float sub)
{
	float offset = 10.; // affects the sharpness of the cusps. offset 0 gives actual cusps with infinitely high hills and low valleys

	// this works...
	// return log(abs(main) + offset) - log(abs(sub) + offset);
	// but it makes the shape a bit strange on both side of the in-between field
	// for example, the field inside of the circle doesn't become fully circular
	// so i guess we have to do it piecewise instead

	// -log(offset) puts one of the objects at elevation 0, but means that offset must not be 0 because then we'd add infinity
	if (main < 0.) // inside the circle
		return log(abs(main) + offset);
	// 	return log(abs(main) + offset) - log(offset);
	else if (sub < 0.) // on the other side of the line
		return - log(abs(sub) + offset);
	else // between the circle and the line
		return log(abs(main) + offset) - log(abs(sub) + offset);
	// 	return log(abs(main) + offset) - (log(abs(sub) + offset) - 3.4);
	// #TODO: why 3.4? i eyeballed it. it makes distancev.x be 0 where main was 0, moving the whole graph surface up or down. the correct number varies with the offset though
}

float combinePerpAsCuspsAtZero(float a, float b)
{
	// i want something similar to min(a, b) but smooth instead of sharp between the zero points

	// caution when i have two line segments at an angle, that share an endpoint
	// if i pass in the raw distances (to infinite line) i get back a distance field that looks like a cross
	// if i want an angle instead, pass in the "capsule" distances to the line segments instead of infinite lines

	// this gives actual cusps (infinite slope) at the 0 points, and a very round hill between them that looks almost like a half circle
	float rawDistPerp = sqrt(abs(a) * abs(b)); // general form pow(abs(a) * abs(b) * abs(c), 1/3) where 3 is the number of distances to combine

	// this gives not cusps but corners (finite slope) at the 0 points, and a parabolic hill between them
	// float rawDistPerp;
	// if (a < 0. && b < 0.)
	// 	rawDistPerp = max(abs(a), abs(b));
	// else if (a < 0.)
	// 	rawDistPerp = a;
	// else if (b < 0.)
	// 	rawDistPerp = b;
	// else // between the two
	// 	rawDistPerp = (abs(a) * abs(b)) / abs(a + b); // no general form found

	return rawDistPerp;
}

float capsule(vec2 d, int i)
{
	if (d.y < 0.)
		d.y = -d.y;
	else if (d.y > uLineLength[i])
		d.y = d.y - uLineLength[i];
	else
		d.y = 0.;

	return length(d);
}

void main()
{
	vec4 mainColor = texture(uMainTex, vTexCoord);
	float waveTotal = 0.0;
	float envelopeTotal = 0.0;
	float numFieldsAdded = 0.;

	// how to combine two distance fields?
	// if we think of the objects as two valleys at 0 elevation,
	// the hill ridge line betwen them will be lower when the valleys are close (fewer contour lines),
	// and higher when the valleys are far away (more contour lines),
	// the contour lines on the hill don't behave like i want
	// they'd repel rather than attract, like electric or magnetic field lines of the same polarity.
	// on the other hand, if one object is a valley at 0 elevation,
	// and the other object is a hill ridgeline at 100 elevation,
	// there will always be the same number of contour lines between them.
	// in other words, the two distance fields should have opposite polarity.
	//
	// however, to get nice control using our slider params,
	// and to get nice movement of the lines,
	// we really want the object valley bottom at 0.
	// so it's impossible to have both objects looking great at the same time.
	//
	// maybe solve that by drawing 2 combined fields:
	// one is a-b
	// and the other is b-a
	// and maybe blend between their rendered output - fade out their amplitude halfway

	if (uNumDistanceFieldTextures > 4 && uAmplitude[0] != 0.0)
	{
		vec4 lineDistance = texture(uDistanceFieldTex[4], vTexCoord);
		vec4 circleDistance = texture(uDistanceFieldTex[0], vTexCoord);

		// angle
		float distanceRatio = lineDistance.x / (abs(lineDistance.x) + (circleDistance.x));
		// float distAlong = mix(lineDistance.y, circleDistance.y, distanceRatio);
		float distAlong = mix(0., lineDistance.y, 1.-distanceRatio) - circleDistance.y * 6.; // *6 looks better, not sure why
		// #TODO: figure out how to make the contour lines join up perfectly at the angle wraparound point
		distAlong *= 0.3; // why is this needed to get the wavelengths to match the non-combined fields?

		float distPerpWithPolarity = combinePerpAsCuspsWithOppositePolarities(circleDistance.x, lineDistance.x);
		float distPerpPreserving0s = combinePerpAsCuspsAtZero(circleDistance.x, lineDistance.x);
		addTwoCrossingWaves(0, distanceRatio, distPerpWithPolarity, distAlong, distPerpPreserving0s, circleDistance.y, waveTotal, envelopeTotal, mainColor);
		numFieldsAdded++;
	}
	if (uNumDistanceFieldTextures > 5 && uAmplitude[4] != 0.0)
	{
		vec2 ceilingDistance = texture(uDistanceFieldTex[4], vTexCoord).xy;
		vec2 wallDistance = texture(uDistanceFieldTex[5], vTexCoord).xy;

		// how to combine two line-segment distance fields that share a corner?
		// i can think of 2-4 ways to combine them

		// 1. one way is smoothmin for the perp, and a gradual turn for the "along"
		// this doesn't look great because either we get a very compressed region around the center diagonal, or all the waves are too tilted
		float distanceRatio = (abs(ceilingDistance.x) / (abs(ceilingDistance.x) + abs(wallDistance.x)));
		float gainFactor = 10. * uExperimental; // arbitrary above 1. at 1, the lines become parallel. higher values compresses lines more and more around the center diagonal
		distanceRatio = gain(distanceRatio, gainFactor);
		float distAlongForFalloff = mix(-ceilingDistance.y, wallDistance.y, distanceRatio);
		const float smoothness = 0.3; // 0 gives the same result as min(), higher values make the join more and more smooth. #TODO: uniform?
		float distPerpSmoothMin = smin(scaleDistancePerp(-ceilingDistance.x), scaleDistancePerp(wallDistance.x), smoothness) * 40.;
		// float distPerpPreserving0s = combinePerpAsCuspsAtZero(-ceilingDistance.x, wallDistance.x);
		float distPerpPreserving0s = combinePerpAsCuspsAtZero(capsule(ceilingDistance, 4), capsule(wallDistance, 5));
		// if (ceilingDistance.x < 0. && wallDistance.x > 0.)
		// 	addTwoCrossingWaves(4, distanceRatio, distPerpSmoothMin, distAlongForFalloff, distPerpPreserving0s, distAlongForFalloff, waveTotal, envelopeTotal, mainColor);

		// 2. another way is to crisscross, use the "perp" distance from one field as the "along" distance for the other
		addTwoCrossingWaves(4, distanceRatio, scaleDistancePerp(-ceilingDistance.x)*40., scaleDistanceAlong(wallDistance.x, 1000., 100.*uExperimental), -ceilingDistance.x, ceilingDistance.y, waveTotal, envelopeTotal, mainColor);
		addTwoCrossingWaves(4, distanceRatio, scaleDistancePerp(wallDistance.x)*40., scaleDistanceAlong(-ceilingDistance.x, 1000., 100.*uExperimental), wallDistance.x, wallDistance.y, waveTotal, envelopeTotal, mainColor);
		// 2.1. a variation of that would be to just draw once, not twice.
		// scale both x and y equally. log() scaling looks too extreme. scaleDistanceAlong looks better
		// but i don't know how to generate a nice alternating pattern in this case
		// addTwoCrossingWaves(4, distanceRatio, scaleDistanceAlong(-ceilingDistance.x, 5000., 100.*uExperimental), scaleDistanceAlong(wallDistance.x, 5000., 100.*uExperimental), ceilingDistance.x, distAlongForFalloff, waveTotal, envelopeTotal, mainColor);
		// addTwoCrossingWaves(4, distanceRatio, scaleDistancePerp(-ceilingDistance.x)*40., scaleDistancePerp(wallDistance.x)*40., ceilingDistance.x, distAlongForFalloff, waveTotal, envelopeTotal, mainColor);
/*
		// 2.2. a variation of that way is to gradually change the slope of the lines, as when a quad is shown in perspective
		// i haven't figured out how to do that yet. maybe it's impossible here - maybe needs to happen out in javascript where we generate the distance field
		if (ceilingDistance.x < 0. && wallDistance.x > 0.)
		{
			// distPerpSmoothMin = scaleDistancePerp(ceilingDistance.y); // along the ceiling corner
			// distPerpSmoothMin = scaleDistancePerp(wallDistance.x); // parallel with the wall corner
			// distPerpSmoothMin = scaleDistancePerp(-ceilingDistance.x); // parallel with the ceiling corner
			// distPerpSmoothMin = scaleDistancePerp(mix(-ceilingDistance.x, wallDistance.y, uExperimental));

			// float t = inverseLerpClamped(0., 400. * uExperimental, wallDistance.y);
			float t = inverseLerpClamped(0., 400. * uExperimental, -ceilingDistance.x);
			// if (wallDistance.y < 0.)
			// 	t = 0.;

			// distPerpSmoothMin = mix(-ceilingDistance.x, wallDistance.y, t);
			// distPerpSmoothMin = scaleDistancePerp(distPerpSmoothMin);
		}
		else if (ceilingDistance.x < 0. && wallDistance.x < 0.)
			distPerpSmoothMin = scaleDistancePerp(wallDistance.x);
		else if (ceilingDistance.x > 0. && wallDistance.x < 0.)
			distPerpSmoothMin = 0.;
		else
			distPerpSmoothMin = scaleDistancePerp(ceilingDistance.x);
		// distPerpSmoothMin = scaleDistancePerp(distPerpSmoothMin);
		// float distPerpPreserving0s = combinePerpAsCuspsAtZero(ceilingDistance.x, wallDistance.x);
		float distPerpPreserving0s = ceilingDistance.x;
		addTwoCrossingWaves(4, distanceRatio, distPerpSmoothMin, distAlong, distPerpPreserving0s, ceilingDistance.y, waveTotal, envelopeTotal, mainColor);
*/
		numFieldsAdded++;
	}

	// i wanted to loop, but then texture() gives a webgl error message, so i copypasted instead
	// #TODO: figure out a way to loop, or at least minimize duplication!
	if (uNumDistanceFieldTextures > 1 && uAmplitude[1] != 0.0)
	{
		vec4 distancev = texture(uDistanceFieldTex[1], vTexCoord);
		float scaledDistPerp = scaleDistancePerp(distancev.x);
		addTwoCrossingWaves(1, 0.5, scaledDistPerp, distancev.y, distancev.x, distancev.y, waveTotal, envelopeTotal, mainColor);
		numFieldsAdded++;
	}
	if (uNumDistanceFieldTextures > 2 && uAmplitude[2] != 0.0)
	{
		vec4 distancev = texture(uDistanceFieldTex[2], vTexCoord);
		float scaledDistPerp = scaleDistancePerp(distancev.x);
		addTwoCrossingWaves(2, 0.5, scaledDistPerp, distancev.y, distancev.x, distancev.y, waveTotal, envelopeTotal, mainColor);
		numFieldsAdded++;
	}
	if (uNumDistanceFieldTextures > 3 && uAmplitude[3] != 0.0)
	{
		vec4 distancev = texture(uDistanceFieldTex[3], vTexCoord);
		// float scaledDistPerp = scaleDistancePerp(distancev.x);
		addTwoCrossingWaves(3, 0.5, distancev.x, distancev.y, distancev.x, distancev.y, waveTotal, envelopeTotal, mainColor);
		numFieldsAdded++;
	}
	if (uNumDistanceFieldTextures > 5 && uAmplitude[5] != 0.0)
	{
		vec4 distancev = texture(uDistanceFieldTex[5], vTexCoord);
		float scaledDistPerp = scaleDistancePerp(distancev.x);
		addTwoCrossingWaves(5, 0.5, scaledDistPerp, distancev.y, distancev.x, distancev.y, waveTotal, envelopeTotal, mainColor);
		numFieldsAdded++;
	}
	if (uNumDistanceFieldTextures > 6 && uAmplitude[6] != 0.0)
	{
		vec4 distancev = texture(uDistanceFieldTex[6], vTexCoord);
		float scaledDistPerp = scaleDistancePerp(distancev.x);
		addTwoCrossingWaves(6, 0.5, scaledDistPerp, distancev.y, distancev.x, distancev.y, waveTotal, envelopeTotal, mainColor);
		numFieldsAdded++;
	}
	if (uNumDistanceFieldTextures > 7 && uAmplitude[7] != 0.0)
	{
		vec4 distancev = texture(uDistanceFieldTex[7], vTexCoord);
		float scaledDistPerp = scaleDistancePerp(distancev.x);
		addTwoCrossingWaves(7, 0.5, scaledDistPerp, distancev.y, distancev.x, distancev.y, waveTotal, envelopeTotal, mainColor);
		numFieldsAdded++;
	}

	float waveBetweenMinus1AndPlus1 = 0.;
	float waveBetween05and20 = 1.;
	// if (numFieldsAdded != 0.)
	// 	waveBetweenMinus1AndPlus1 = waveTotal / float(numFieldsAdded);
	// 	waveBetween05and20 = (numFieldsAdded == 0.) ? 1. : reMap(waveTotal / float(numFieldsAdded), -1., 1., 0.5, 1.5);
	if (envelopeTotal != 0.)
		waveBetweenMinus1AndPlus1 = waveTotal / envelopeTotal;

	//fragColor = vec4(mainColor.rgb * waveBetween05and20, 1.0);
	fragColor = vec4(mainColor.rgb + uOpacity * waveBetweenMinus1AndPlus1, 1.0);
	// if (waveTotal < 0.)
	// 	fragColor = vec4(0., 1., 0., 1.0); // never happens
}
