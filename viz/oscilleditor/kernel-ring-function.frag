// this file will be #included in other files via our homemade include directive

// Define characteristic distances and widths for each lever
uniform float kernelMaxDistance; // 5 still feels realtime, 10 feels too slow. #TODO: optimize somehow. try precalculating kernel? or decompose? see https://bartwronski.com/2020/02/03/separate-your-filters-svd-and-low-rank-approximation-of-image-filters/ or https://www.intel.com/content/www/us/en/developer/articles/technical/an-investigation-of-fast-real-time-gpu-based-image-blur-algorithms.html


// NOTE: these ring kernel functions are implemented in both javascript and glsl shader kernel-ring-function.frag,
// so if you change one, remember to change the other to match!
// #TODO: performance is a problem here, but there is a trick to speed up this kind of convolution 
// the gaussian is separable, meaning you can do a two-pass thing in O(2N) instead of one-pass O(N^2)
// https://gamedev.stackexchange.com/questions/26649/glsl-one-pass-gaussian-blur
// https://stackoverflow.com/questions/5243983/what-is-the-most-efficient-way-to-implement-a-convolution-filter-within-a-pixel

/*
// old version, from when kernelRingWidths and friends were arrays of size 4, not vec4

uniform float kernelRingCoupling[4];
uniform float kernelRingDistances[4]; // as fractions of max distance
uniform float kernelRingWidths[4]; // as fractions of max distance

float ringFunction(float distance, float center, float width)
{
	float innerGaussian = 0.0;//exp(-pow(distance - center, 2.0) / (2.0 * pow(width * 0.5, 2.0)));
	float outerGaussian = exp(-pow(distance - center, 2.0) / (2.0 * pow(width, 2.0)));
	return innerGaussian - outerGaussian;
}

float influenceFunction(float distance, float maxDistance)
{
	float totalInfluence = 0.0;

	for (int i = 0; i < 4; i++)
	{
		float r_i = kernelRingDistances[i] * maxDistance;
		float width_i = kernelRingWidths[i] * maxDistance;
		float A_i = -kernelRingCoupling[i];

		totalInfluence += A_i * ringFunction(distance, r_i, width_i);
	}

	return totalInfluence;
}

float influenceFunction2(float distance, float inverseMaxDistance)
{
	distance *= inverseMaxDistance;
	float totalInfluence = 0.0;

	for (int i = 0; i < 4; i++)
	{
		float r_i = kernelRingDistances[i];
		float width_i = kernelRingWidths[i];
		float A_i = -kernelRingCoupling[i];

		totalInfluence += A_i * ringFunction(distance, r_i, width_i);
	}

	return totalInfluence;
}
*/

// new version, slightly faster, speed increased from 727 frames per minute to 803 frames per minute in a test run with a large kernel
uniform vec4 kernelRingCoupling;
uniform vec4 kernelRingDistances; // as fractions of max distance
uniform vec4 kernelRingWidths; // as fractions of max distance
// i think that kernelRingDistances[3] + kernelRingWidths[3]/2 must be <= 1 otherwise we get a sharp edge of the convolution kernel?

float influenceFunction(float sampleDistanceFromOrigo, float maxDistance)
{
	vec4 sampleDistancesFromRingPeak = sampleDistanceFromOrigo - kernelRingDistances * maxDistance;

	vec4 totalInfluences = kernelRingCoupling * exp(
		-(sampleDistancesFromRingPeak * sampleDistancesFromRingPeak)
		/
		(2.0 * kernelRingWidths * kernelRingWidths * maxDistance * maxDistance)
	);

	float totalInfluence = totalInfluences.x + totalInfluences.y + totalInfluences.z + totalInfluences.w;
	// float totalInfluence = dot(totalInfluences, vec4(1.0)); // maybe faster?

	return totalInfluence;
}

// probably faster than influenceFunction(), because we reduce 8 multiplications to 1 multiplication
// but requires that the caller takes the inverse of maxDistance, and divisions are costly,
// so probably only worth doing if done once, outside the convolution loop
float influenceFunction2(float sampleDistanceFromOrigo, float inverseMaxDistance)
{
	sampleDistanceFromOrigo *= inverseMaxDistance;

	vec4 sampleDistancesFromRingPeak = sampleDistanceFromOrigo - kernelRingDistances;

	vec4 totalInfluences = kernelRingCoupling * exp(
		-(sampleDistancesFromRingPeak * sampleDistancesFromRingPeak)
		/
		(2.0 * kernelRingWidths * kernelRingWidths)
	);

	float totalInfluence = totalInfluences.x + totalInfluences.y + totalInfluences.z + totalInfluences.w;
	// float totalInfluence = dot(totalInfluences, vec4(1.0)); // maybe faster?

	return totalInfluence;
}
