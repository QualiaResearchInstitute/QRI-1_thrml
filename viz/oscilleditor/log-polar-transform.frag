// this file will be #included in other files via our homemade include directive

// our log-polar coordinate transform feature, aka "form constant", interacts with a lot of other effects
// we have to use the same transform when setting oscillator frequencies and kernel size multipliers
// and be careful when coupling between layers, if one layer uses the transform and the other layer doesn't
// actually different frequency settings should behave differently
// for "Spatial sine&cosine pattern" we do it before transform, so you get a nice typical form constant
// but for the Depthmap/Edgemap, we sample from the transformed position in the oscillator field, to the untransformed location in the source image
// which means to loop through the oscillator field in x and y and _inverse_ transform them to tx and ty, and sample in the source image at tx and ty

// mathematically, these functions are inverse of each other.
// but when we use them to sample from a texture, and polarRadius ends up outside the range [0,1] it will wrap around as usual thanks to texParameteri REPEAT
// so for example 2.3 will wrap around twice and end up at 0.3
// which means the functions are no longer bijective.
// but the constant 23.0 was chosen (by trial and error) so that there is as little wraparound as possible.
// the circular outer edge juuuust touches the four corners of the canvas rectangle.
// there is wraparound near the center, but very small
// the constant 0.2 affects what happens near the center. it could have been 1.0 which would have meant no wraparound, but a lot of visible distortion.
// or it could have been 0.0 which would have meant no distortion, but infinite wraparound, polarRadius would approach negative infinity
// so 0.2 was a compromise

// must be the inverse of doLogPolarTransform, i.e. doLogPolarInverseTransform(wh, doLogPolarTransform(wh, uv)) == uv
vec2 doLogPolarInverseTransform(vec2 texWidthHeight, vec2 polar)
{
	float invAspectRatio = texWidthHeight.y / texWidthHeight.x;
	float polarAngle0To1 = polar.x;
	float polarAngle0To2Pi = polarAngle0To1 * TAU;
	float polarAngleMinusPiToPi = polarAngle0To2Pi - PI;
	float polarRadius = polar.y;
	float length_apa = (exp(polarRadius * PI) - 0.2) / 23.0;
	vec2 apa = vec2(cos(polarAngleMinusPiToPi), sin(polarAngleMinusPiToPi)) * length_apa;
	apa.x *= invAspectRatio;
	vec2 uv = apa + vec2(0.5, 0.5);
	return uv;
}

// must be the inverse of doLogPolarInverseTransform, i.e. doLogPolarTransform(wh, doLogPolarInverseTransform(wh, uv)) == uv
vec2 doLogPolarTransform(vec2 texWidthHeight, vec2 uv)
{
	float aspectRatio = texWidthHeight.x / texWidthHeight.y;
	vec2 apa = uv - vec2(0.5, 0.5); // apa is 00 in the screen center
	apa.x *= aspectRatio; // otherwise the tunnel gets an elliptic shape instead of circular
	float polarRadius = log(0.2 + length(apa) * 23.0) / PI;
	float polarAngleMinusPiToPi = atan(apa.y, apa.x);
	float polarAngle0To2Pi = polarAngleMinusPiToPi + PI;
	float polarAngle0To1 = polarAngle0To2Pi / TAU;
	vec2 polar = vec2(polarAngle0To1, polarRadius);
	return polar;
}
