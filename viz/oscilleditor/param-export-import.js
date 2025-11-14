function tryMakeUrl(jsonString, baseUrlWithQuestionMark) {
	let parameters;
	try {
		parameters = JSON.parse(jsonString);
	}
	catch (error) {
		console.log(jsonString);
		console.error(error);
		return false;
	}

	// option 1: query string as submitted by a classic html <form>, formatted like ?foo=344&bar=false&baz=rainbow
	const reSerialized = urlQueryStringFromObject(parameters);
/*
	// option 2: serialize to JSON
	// but the URLs become very long
	// try to shorten them a bit, to stay under 2000 characters
	Object.keys(parameters).forEach(key => {
		// is this a string but could be a number instead? saves 2 characters of length (the quote marks)
		if (typeof parameters[key] == "string") {
			const regex = /^[-.0-9]+$/;
			if (regex.test(parameters[key])) {
				const asNumber = +parameters[key];
				const backToString = ""+asNumber;
				// console.log(parameters[key], asNumber, backToString);
				if (parameters[key] === backToString) {
					parameters[key] = asNumber;
				}
 			}
		}
	});
	parameters = compressParamsForJson(parameters);
	const reSerialized = JSON.stringify(parameters);
*/
	// console.log(reSerialized);
	if (baseUrlWithQuestionMark.substr(baseUrlWithQuestionMark.length-1, 1) != "?")
		baseUrlWithQuestionMark += "?";
	return baseUrlWithQuestionMark + encodeURI(reSerialized);
}

function urlQueryStringFromObject(jsonObj) {
	// const keyValuePairs = [];
	// for (let key in jsonObj) {
	// 	keyValuePairs.push(key + "=" + jsonObj[key]);
	// }
	// return keyValuePairs.join("&");

	jsonObj = compressParamsForQueryString(jsonObj);
	const searchParams = new URLSearchParams(jsonObj);
	return searchParams.toString();
}

function objectFromUrlQueryString(queryString) {
	const numericalStringRegex = /^[-.0-9]+$/;
	const searchParamsTmp = new URLSearchParams(queryString);
	// console.log(searchParamsTmp);
	// convert to normal javascript object
	let searchParams = {};
	for (const [key, value] of searchParamsTmp)
		searchParams[key] = value;
	// console.log(searchParams);
	searchParams = uncompressParamsForQueryString(searchParams);
	// console.log(searchParams);
	const result = {};
	// for (const [key, value] of searchParams) {
	for (let key in searchParams) {
		const value = searchParams[key];
		// this function doesn't know for sure if the string "5" should be converted to the number 5 or left as-is
		// but luckily it doesn't matter for our purposes. let's just convert them all
		if (value == "false")
			result[key] = false;
		else if (value == "true")
			result[key] = true;
		else if (numericalStringRegex.test(value))
			result[key] = parseFloat(value);
		else
			result[key] = value;
	}
	// console.log(result);
	return result;
}

function compressParamsForQueryString(full) {
	const compressed = {};
	const layerRingRegex = /^(layer[0-9]+ring)([0-9]+)/;
	const layerOsciFreqRegex = /^layer[0-9]+oscillatorFrequency/;
	const numericalStringRegex = /^[-.0-9]+$/;
	for (let key in full) {
		let value = full[key];
		if (numericalStringRegex.test(value))
			value = parseFloat(value);

		let searchResult;
		if (searchResult = key.match(layerRingRegex)) {
			const restOfKey = key.substr(searchResult[0].length);
			const ringIndex = parseInt(searchResult[2]);
			const newKey = searchResult[1] + restOfKey; // from "layer0ring2coupling" to "layer0ringcoupling"
			setKeyIfNotExists(compressed, newKey, []);
			compressed[newKey][ringIndex] = value;
		}
		else if (searchResult = key.match(layerOsciFreqRegex)) {
			const restOfKey = key.substr(searchResult[0].length);
			const minMaxIndex = (restOfKey == "Min" ? 0 : 1);
			setKeyIfNotExists(compressed, searchResult[0], []);
			compressed[searchResult[0]][minMaxIndex] = value;
		}
		else
			compressed[key] = value;
	}

	for (let key in compressed) {
		if (Array.isArray(compressed[key]))
			compressed[key] = compressed[key].join("_");
	}

	return compressed;
}

function uncompressParamsForQueryString(compressed) {
	// console.log(compressed);
	const full = {};
	const layerRingRegex = /^(layer[0-9]+ring)[^0-9]/;
	const layerOsciFreqRegex = /^layer[0-9]+oscillatorFrequency$/;
	const serializedArrayRegex = /^[-0-9._]+$/;
	// for (const [key, value] of compressed) {
	for (let key in compressed) {
		const value = compressed[key];
		let searchResult;
		searchResult = key.match(layerRingRegex);
		if ((searchResult = key.match(layerRingRegex)) && value.match(serializedArrayRegex)) {
			const values = value.split("_");
			const restOfKey = key.substr(searchResult[1].length);
			// console.log(key, value, searchResult, values, restOfKey);
			for (let ringIndex in values)
				full[searchResult[1] + ringIndex + restOfKey] = values[ringIndex];
		}
		else if ((searchResult = key.match(layerOsciFreqRegex)) && value.match(serializedArrayRegex)) {
			const values = value.split("_");
			// console.log(key, value, searchResult, values);
			full[key + "Min"] = values[0];
			full[key + "Max"] = values[1];
		}
		else {
			// console.log(key, value);
			full[key] = value;
		}
	}
	// console.log(full);
	return full;
}
/*
// change the structure from a flat array to a nested thing with sub-arrays, see unit tests below for examples
function compressParamsForJson(full) {
	const compressed = {};
	const layerRegex = /^layer([0-9]+)/;
	const ringRegex = /^ring([0-9]+)/;
	for (let key in full) {
		let searchResult;
		if (searchResult = key.match(layerRegex)) {
			const layerIndex = parseInt(searchResult[1]);
			const restOfKey = key.substr(searchResult[0].length);
			setKeyIfNotExists(compressed, "layers", []);
			setKeyIfNotExists(compressed["layers"], layerIndex, {});

			if (searchResult = restOfKey.match(ringRegex)) {
				const ringIndex = parseInt(searchResult[1]);
				const keyWithoutIndex = "ring" + restOfKey.substr(searchResult[0].length);
				setKeyIfNotExists(compressed["layers"][layerIndex], keyWithoutIndex, []);
				compressed["layers"][layerIndex][keyWithoutIndex][ringIndex] = full[key];
			}
			else if (restOfKey.startsWith("oscillatorFrequency")) {
				setKeyIfNotExists(compressed["layers"][layerIndex], "oscillatorFrequency", []);
				const minMaxIndex = (restOfKey == "oscillatorFrequencyMin" ? 0 : 1);
				compressed["layers"][layerIndex]["oscillatorFrequency"][minMaxIndex] = full[key];
			}
			else
				compressed["layers"][layerIndex][restOfKey] = full[key];
		}
		else
			compressed[key] = full[key];
	}

	return compressed;
}

function uncompressParamsForJson(compressed) {
	const full = {};
	for (let key in compressed) {
		if (key == "layers") {
			for (let layerIndex in compressed[key])
				for (let keyInLayer in compressed[key][layerIndex]) {
					if (keyInLayer.startsWith("ring") && Array.isArray(compressed[key][layerIndex][keyInLayer])) {
						for (let ringIndex in compressed[key][layerIndex][keyInLayer]) {
							const keyWithRingIndex = "ring" + ringIndex + keyInLayer.substr("ring".length);
							full["layer" + layerIndex + keyWithRingIndex] = compressed[key][layerIndex][keyInLayer][ringIndex];
						}
					}
					else if (keyInLayer.startsWith("oscillatorFrequency") && Array.isArray(compressed[key][layerIndex][keyInLayer])) {
						full["layer" + layerIndex + keyInLayer + "Min"] = compressed[key][layerIndex][keyInLayer][0];
						full["layer" + layerIndex + keyInLayer + "Max"] = compressed[key][layerIndex][keyInLayer][1];
					}
					else
						full["layer" + layerIndex + keyInLayer] = compressed[key][layerIndex][keyInLayer];
				}
		}
		else
			full[key] = compressed[key];
	}
	return full;
}
*/
function setKeyIfNotExists(obj, key, value) {
	if (typeof obj[key] == "undefined")
		obj[key] = value;
}

const fullParams = {
  "photoIndex": 5,
  "driftMaxDisplacement": "0.010",
  "driftPatternPlaybackSpeed": "4",
  "driftPatternSize": "1280",
  "deepDreamLayerIndex": "0",
  "useCustomVideo": false,
  "customVideoPlaybackSpeed": "0.5",
  "deepDreamOpacity": "0.0",
  "deepLuminosityBlendOpacity": "0.5",
  "blendPatternPlaybackSpeed": "0.2",
  "blendPatternSize": "1280",
  "oscillatorOpacity": "0.25",
  "oscillatorColors": "rainbow",
  "numOscillatorFields": "1",
  "crossFieldBlendingBalance": "1.5",
  "layer0logPolar": false,
  "layer0fieldResolutionHalvings": "0",
  "layer0kernelMaxDistance": "5",
  "layer0kernelSizeShrunkBy": "nothing",
  "layer0kernelShrinkFactor": "0.3",
  "layer0ring0coupling": "1.0",
  "layer0ring0distance": "0.1",
  "layer0ring0width": "0.05",
  "layer0ring1coupling": "1.0",
  "layer0ring1distance": "0.3",
  "layer0ring1width": "0.08",
  "layer0ring2coupling": "-1.0",
  "layer0ring2distance": "0.6",
  "layer0ring2width": "0.12",
  "layer0ring3coupling": "0.0",
  "layer0ring3distance": "0.9",
  "layer0ring3width": "0.15",
  "layer0influenceFromOtherLayer": "0.0",
  "layer0oscillatorFrequencyMin": "1",
  "layer0oscillatorFrequencyMax": "3",
  "layer0frequencyCorrelationPatternSize": "100",
  "layer1logPolar": true,
  "layer1fieldResolutionHalvings": "1",
  "layer1kernelMaxDistance": "10",
  "layer1kernelSizeShrunkBy": "nothing",
  "layer1kernelShrinkFactor": "0.3",
  "layer1ring0coupling": "0.0",
  "layer1ring0distance": "0.1",
  "layer1ring0width": "0.05",
  "layer1ring1coupling": "0.0",
  "layer1ring1distance": "0.3",
  "layer1ring1width": "0.08",
  "layer1ring2coupling": "0.0",
  "layer1ring2distance": "0.6",
  "layer1ring2width": "0.12",
  "layer1ring3coupling": "0.0",
  "layer1ring3distance": "0.9",
  "layer1ring3width": "0.15",
  "layer1influenceFromOtherLayer": "0.0",
  "layer1oscillatorFrequencyMin": "3",
  "layer1oscillatorFrequencyMax": "7",
  "layer1frequencyCorrelationPatternSize": "100"
};

const compressedQueryString = "photoIndex=5&driftMaxDisplacement=0.01&driftPatternPlaybackSpeed=4&driftPatternSize=1280&deepDreamLayerIndex=0&useCustomVideo=false&customVideoPlaybackSpeed=0.5&deepDreamOpacity=0&deepLuminosityBlendOpacity=0.5&blendPatternPlaybackSpeed=0.2&blendPatternSize=1280&oscillatorOpacity=0.25&oscillatorColors=rainbow&numOscillatorFields=1&crossFieldBlendingBalance=1.5&layer0logPolar=false&layer0fieldResolutionHalvings=0&layer0kernelMaxDistance=5&layer0kernelSizeShrunkBy=nothing&layer0kernelShrinkFactor=0.3&layer0ringcoupling=1_1_-1_0&layer0ringdistance=0.1_0.3_0.6_0.9&layer0ringwidth=0.05_0.08_0.12_0.15&layer0influenceFromOtherLayer=0&layer0oscillatorFrequency=1_3&layer0frequencyCorrelationPatternSize=100&layer1logPolar=true&layer1fieldResolutionHalvings=1&layer1kernelMaxDistance=10&layer1kernelSizeShrunkBy=nothing&layer1kernelShrinkFactor=0.3&layer1ringcoupling=0_0_0_0&layer1ringdistance=0.1_0.3_0.6_0.9&layer1ringwidth=0.05_0.08_0.12_0.15&layer1influenceFromOtherLayer=0&layer1oscillatorFrequency=3_7&layer1frequencyCorrelationPatternSize=100";

// console.log(compressedQueryString);
// console.log(objectFromUrlQueryString(compressedQueryString));
// console.log(urlQueryStringFromObject(objectFromUrlQueryString(compressedQueryString)));
console.assert(urlQueryStringFromObject(objectFromUrlQueryString(compressedQueryString)) == compressedQueryString);

// console.log(fullParams);
// console.log(urlQueryStringFromObject(fullParams));
// console.log(objectFromUrlQueryString(urlQueryStringFromObject(fullParams)));
// console.assert(doObjectsHaveSameKeysAndValues(objectFromUrlQueryString(urlQueryStringFromObject(fullParams)), fullParams)); // no, fails because we converted strings to floats

/*
const compressedJsonParams = {
	"photoIndex": 5,
	"driftMaxDisplacement": "0.010",
	"driftPatternPlaybackSpeed": "4",
	"driftPatternSize": "1280",
	"deepDreamLayerIndex": "0",
	"useCustomVideo": false,
	"customVideoPlaybackSpeed": "0.5",
	"deepDreamOpacity": "0.0",
	"deepLuminosityBlendOpacity": "0.5",
	"blendPatternPlaybackSpeed": "0.2",
	"blendPatternSize": "1280",
	"oscillatorOpacity": "0.25",
	"oscillatorColors": "rainbow",
	"numOscillatorFields": "1",
	"crossFieldBlendingBalance": "1.5",
	"layers": [
		{
			"logPolar": false,
			"fieldResolutionHalvings": "0",
			"kernelMaxDistance": "5",
			"kernelSizeShrunkBy": "nothing",
			"kernelShrinkFactor": "0.3",
			"ringcoupling": ["1.0", "1.0", "-1.0", "0.0"],
			"ringdistance": ["0.1", "0.3", "0.6", "0.9"],
			"ringwidth": ["0.05", "0.08", "0.12", "0.15"],
			"influenceFromOtherLayer": "0.0",
			"oscillatorFrequency": ["1", "3"],
			"frequencyCorrelationPatternSize": "100",
		},
		{
			"logPolar": true,
			"fieldResolutionHalvings": "1",
			"kernelMaxDistance": "10",
			"kernelSizeShrunkBy": "nothing",
			"kernelShrinkFactor": "0.3",
			"ringcoupling": ["0.0", "0.0", "0.0", "0.0"],
			"ringdistance": ["0.1", "0.3", "0.6", "0.9"],
			"ringwidth": ["0.05", "0.08", "0.12", "0.15"],
			"influenceFromOtherLayer": "0.0",
			"oscillatorFrequency": ["3", "7"],
			"frequencyCorrelationPatternSize": "100"
		}
	]
};

// console.assert(JSON.stringify(compressParamsForJson(fullParams)) == JSON.stringify(compressedJsonParams));
// console.assert(JSON.stringify(compressParamsForJson(uncompressParamsForJson(compressedJsonParams))) == JSON.stringify(compressedJsonParams));
// console.assert(doObjectsHaveSameKeysAndValues(uncompressParamsForJson(compressedJsonParams), fullParams));
// // console.assert(JSON.stringify(uncompressParamsForJson(compressedJsonParams)) == JSON.stringify(fullParams)); // no, bad test because the order might be different, which is OK
*/
function doObjectsHaveSameKeysAndValues(obj1, obj2) {
	// return JSON.stringify(obj1) == JSON.stringify(obj2); // no, bad test because the order might be different, which is OK
	if (!obj1)
		return false;
	if (!obj2)
		return false;
	if (Object.keys(obj1).length !== Object.keys(obj2).length)
		return false;
	for (let i in obj1) {
		if (obj1[i] !== obj2[i])
			return false;
	}
	return true;
}
// console.assert(doObjectsHaveSameKeysAndValues({x:1,y:2}, {x:1,y:2}));
// console.assert(doObjectsHaveSameKeysAndValues({x:1,y:2}, {y:2,x:1}));
// console.assert(!doObjectsHaveSameKeysAndValues({x:1,y:2}, {x:1,y:3}));
// console.assert(!doObjectsHaveSameKeysAndValues({x:1,y:2}, {x:1,z:2}));
