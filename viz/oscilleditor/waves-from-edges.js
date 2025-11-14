// global variables have the g_ prefix
let g_animFrameId;
let g_simulateSingleStep = false;
let g_isPlaying = false;
let g_performanceTimerStart;
let g_numFramesThisSecond = 0;
let g_numFrames = 0;
let g_estimatedFps = 60; // will get estimated after 1 second of runtime

let g_debugDrawSomeWave = 0;
let g_debugDrawDfIndex = 0;
let g_debugDrawWaveIndex = 0;
let g_debugDrawSomeWaveForNFrames = 60;

// web gl objects
let g_webglContext;
let g_quasicrystalShader;
let g_oscillatorFramebuffer;

// assets
let g_mainImage; // #TODO: sourcePhoto or baseImage is a better term
let g_mainImageTexture;
let g_distanceFieldTextures = [];
let g_lineLengths = [];

// configuration, adjustable by the user
// many are passed directly into the shader as uniforms, but not all
// see serializeConfigStateToUrl and deserializeConfigStateFromUrl
let g_params = {
	// this object will be filled with a lot more properties too, dynamically, based on the input fields in the html
};

if (document.readyState !== 'loading')
	setup();
else
	document.addEventListener('DOMContentLoaded', setup);


function setErrorMessage(msg) {
	console.error(msg);
	const paragraph = document.getElementById("glCanvasMessage");
	paragraph.textContent = msg;
}

// import { SVD } from 'svd-js'

async function setup() {

	// #TODO: https://developer.mozilla.org/en-US/docs/Web/API/WebGL_API/WebGL_best_practices

	const canvas = document.getElementById("glCanvas");
	g_webglContext = canvas.getContext("webgl2");
	if (!g_webglContext) {
		setErrorMessage("Failed to get WebGL2 context. Your browser or device probably doesn't support WebGL 2.");
		return;
	}

	// #TODO: Handle context lost
	// canvas.addEventListener('webglcontextlost', function(event) {
	// 	console.warn('webglcontextlost');
	// 	event.preventDefault();
	// }, false);

	// console.log(g_webglContext.getExtension('OES_texture_float')); // enabled in WebGL 2 by default

	// enable extension that allow us to write to floating-point textures (but only RGBA32F, sadly not RGB32F)
	// https://developer.mozilla.org/en-US/docs/Web/API/WEBGL_color_buffer_float
	// https://developer.mozilla.org/en-US/docs/Web/API/EXT_color_buffer_float
	// if (g_webglContext.getExtension('WEBGL_color_buffer_float')) { // for WebGL 1
	if (g_webglContext.getExtension('EXT_color_buffer_float') == null) { // for WebGL 2
		setErrorMessage("Failed to enable the WebGL 2 extension EXT_color_buffer_float. Your browser or device probably doesn't support it.");
		// #TODO: gracefully fall back to use unsigned byte texture instead?
		return;
	}

	// enable extension that allows us to filter floating-point textures, ie choose gl.LINEAR instead of just gl.NEAREST
	// https://developer.mozilla.org/en-US/docs/Web/API/OES_texture_float_linear
	if (g_webglContext.getExtension('OES_texture_float_linear') == null) {
		setErrorMessage("Failed to enable the WebGL 2 extension OES_texture_float_linear. Your browser or device probably doesn't support it.");
		// #TODO: gracefully fall back to use unsigned byte texture instead?
		return;
	}



	// fire off a bunch of downloads as Promises, in parallel
	const quasicrystalShaderPromise = loadShader(g_webglContext, 'shader.vert', 'waves-from-edges.frag');
	// Load images and videos from 'assets' folder
	const tinyDummyImagePromise = downloadImage('assets/', 'white1px.png', true);
	// do other stuff while waiting for the downloads


	const sourcePhotos = [
		{
			base: "room.jpg",
			// numDistanceFields: 6,
		},
	];

	function queueMainImageDownload(sourcePhotoFilenames) {
		const mainImageDownloader = {};
		mainImageDownloader.sourcePhotoFilenames = sourcePhotoFilenames;
		mainImageDownloader.mainImagePromise = downloadImage('assets/', sourcePhotoFilenames.base, true);
		// mainImageDownloader.distanceFieldPromises = [];
		// for (let i = 0; i < sourcePhotoFilenames.numDistanceFields; i++) {
		// 	let filename = sourcePhotoFilenames.base;
		// 	filename = filename.substr(0, filename.lastIndexOf(".")) + "_" + (i+1) + ".png";
		// 	// console.log(filename);
		// 	mainImageDownloader.distanceFieldPromises.push(downloadImage('assets/distancefields/', filename));
		// }
		return mainImageDownloader;
	}

	// must be done _after_ all the input fields exist in the html page
	// but must be done _before_ we pick sourcePhotos[i]
	// wait a minute, that's impossible
	// most of the input fields always exist, but deepDreamLayerIndex is dynamically created based on sourcePhotos[i]
	// ok #TODO: i guess we need to do it in two steps
	// if (window.location.search)
	// 	deserializeConfigStateFromUrl();

	// fire off a bunch of downloads as Promises, in parallel
	// Load images and videos from 'assets' folder
	const mainImageDownloader = queueMainImageDownload(sourcePhotos[0]);
	// then await until the item is downloaded and generate texture
	// do other stuff meanwhile





	// Add event listeners to HTML controls

	function updatePlayPauseButtons() {
		document.getElementById('btnPlay').disabled = g_isPlaying;
		document.getElementById('btnPause').disabled = !g_isPlaying;
		document.getElementById('btnSingleStep').disabled = g_isPlaying;
	}
	updatePlayPauseButtons();
	document.getElementById('btnPause').addEventListener('click', (e) => {
		g_isPlaying = false;
		updatePlayPauseButtons();
	});
	document.getElementById('btnPlay').addEventListener('click', (e) => {
		if (!g_isPlaying) {
			g_isPlaying = true;
			updatePlayPauseButtons();
			g_performanceTimerStart = performance.now();
			g_numFramesThisSecond = 0;
			requestAnimationFrameNoDuplicate();
		}
	});
	document.getElementById('btnSingleStep').addEventListener('click', (e) => {
		if (!g_isPlaying) {
			g_simulateSingleStep = true;
			requestAnimationFrameNoDuplicate();
		}
	});


	// any change to input form parameters means:
	// we should store the new value in the g_params object (for faster lookup during rendering, and easy exporting to JSON)
	// we should re-render
	// beyond that, it's different for different parameters
	const inputs = document.querySelectorAll('input, select');
	inputs.forEach(input => {
		if (input.id || (input.name && input.type == 'radio')) {

			// all our sliders have a number input field beside them, and they should stay in sync both ways
			let associatedNumberInput;
			let associatedSliderInput;
			if (input.type === 'range')
				associatedNumberInput = document.getElementById(input.id + 'Value');
			if (input.type === 'number')
				associatedSliderInput = document.getElementById(input.id.replace(/Value$/, ''));
			if (associatedNumberInput) {
				// Set number input attributes to match the slider
				associatedNumberInput.min = input.min;
				associatedNumberInput.max = input.max;
				associatedNumberInput.step = input.step;
				associatedNumberInput.value = input.value;
			}

			// find a canonical parameter name
			// note that we give the same canonical name to the slider and the associated number input
			let paramName;
			if (input.type === 'checkbox')
				paramName = input.id;
			else if (input.type === 'radio')
				paramName = input.name; // not input.id! a bunch of radio buttons with the same name represent the same parameter
			else if (['range', 'text', 'number', 'select-one'].includes(input.type))
				paramName = (associatedSliderInput ? associatedSliderInput.id : input.id); // don't use numOscillatorFieldsValue, use numOscillatorFields

			// identify special cases based on param names
			let paramChangeShouldDebugDrawWave = false;
			let paramIsSpeed = false;
			let paramIsWavelength = false;
			let paramIsPeriod = false;
			let paramIsFrequency = false;
			let iWave = -1;
			let wavePartOfName;
			let restOfName;
			{
				const waveRegexp = /^wave([0-9])/g;
				const matches = waveRegexp.exec(paramName);
				if (matches) {
					iWave = parseInt(matches[1]);
					restOfName = paramName.substr(waveRegexp.lastIndex);
					wavePartOfName = paramName.substr(0, waveRegexp.lastIndex);
					// console.log(paramName, iWave, waveRegexp.lastIndex, wavePartOfName, restOfName);
					paramChangeShouldDebugDrawWave = true;

					if (restOfName == "Speed") {
						paramIsSpeed = true;
					}
					else if (restOfName == "Wavelength") {
						paramIsWavelength = true;
					}
					else if (restOfName == "Period") {
						paramIsPeriod = true;
					}
					else if (restOfName == "Frequency") {
						paramIsFrequency = true;
					}

					// conditions for when layer inputs should be enabled / disabled
					// .enabledCondition is not a thing that already exists on DOM objects, we just invent it and add it on our own
					// the enabledCondition function will be called later
					// if (restOfName == 'kernelShrinkFactor')
					// 	input.enabledCondition = function() {
					// 		return (g_params['layer'+iEdge+'kernelSizeShrunkBy'] != 'nothing') && isNumActiveLayersGreaterThan(iEdge);
					// 	};
					// else if (restOfName == 'frequencyCorrelationPatternSize')
					// 	input.enabledCondition = function() {
					// 		return (g_params['layer'+iEdge+'frequenciesVaryBy'] == 'pattern') && isNumActiveLayersGreaterThan(iEdge);
					// 	};
					// else
					// 	input.enabledCondition = function() { return isNumActiveLayersGreaterThan(iEdge); };
				}
			}

			// conditions for when inputs should be enabled / disabled
			// if (paramName == 'customVideoPlaybackSpeed')
			// 	input.enabledCondition = function() {
			// 		return g_params.useCustomVideo;
			// 	};
			// else if (paramName == 'crossFieldBlendingBalance')
			// 	input.enabledCondition = function() { return isNumActiveLayersGreaterThan(1); };
			// else if (paramName == 'layer0influenceFromOtherLayer')
			// 	input.enabledCondition = function() { return isNumActiveLayersGreaterThan(1); };

			// console.log(input.type, input.id, input.name, restOfName, wavePartOfName, iWave, paramChangeShouldDebugDrawWave, paramIsSpeed, paramIsWavelength, paramIsPeriod, paramIsFrequency);

			function storeHtmlInputValueInGlobalParamsObject() {
				let newParamValue;
				if (input.type === 'checkbox')
					newParamValue = input.checked;
				else if (input.type === 'radio' && input.checked)
					// only grab the value from the _checked_ radio button. the others non-checked radios return undefined
					newParamValue = input.value;
				else if (['number'].includes(input.type)) {
					if (!isNaN(parseFloat(input.value)))
						newParamValue = input.value;
					// else leave newParamValue undefined so that we don't update g_params, in other words g_params will remember the previous valid value
				}
				else if (['range', 'text', 'number', 'select-one'].includes(input.type))
					newParamValue = input.value;

				if (newParamValue !== undefined) {
					// Synchronize sliders and number inputs
					if (associatedNumberInput)
						associatedNumberInput.value = input.value;
					else if (associatedSliderInput)
						associatedSliderInput.value = input.value;

					if (g_params[paramName] != newParamValue) { // because we often get hit with 'input' and 'change' in quick succession
						g_params[paramName] = newParamValue;

						// update enabled / disabled state for ALL inputs, it's easiest, so we don't have to track dependencies
						inputs.forEach(input => {
							if (input.enabledCondition)
								input.disabled = !input.enabledCondition();
						});

						return true;
					}
				}
				return false;
			}

			// first do this once, at startup:
			storeHtmlInputValueInGlobalParamsObject();
			// then add event listeners to do it every time the user changes the param

			function setParamAndHtmlFieldValue(id, newValue) {
				// console.log("id", newValue);
				g_params[id] = newValue;
				const htmlInputField = document.getElementById(id);
				if (htmlInputField) {
					htmlInputField.value = newValue;
					document.getElementById(id+'Value').value = newValue;
				}
			}

			function storeAndReRender() {
				// console.log("storeAndReRender", input);
				if (storeHtmlInputValueInGlobalParamsObject()) {

					// I made an arbitrary choice here that the speed is never auto-changed
					// so for example in the equation wavelength/speed=period
					// if you change the wavelength, i keep the speed fixed and auto-change the period, not the other way around
					if (paramIsSpeed || paramIsWavelength) {
						const speed = parseFloat(g_params[wavePartOfName+'Speed']);
						const wavelength = parseFloat(g_params[wavePartOfName+'Wavelength']);
						if (!isNaN(speed) && !isNaN(wavelength)) {
							const period = wavelength / speed;
							const frequency = 1 / period;
							setParamAndHtmlFieldValue(wavePartOfName+'Period', period);
							setParamAndHtmlFieldValue(wavePartOfName+'Frequency', frequency);
						}
					}
					if (paramIsPeriod) {
						const speed = parseFloat(g_params[wavePartOfName+'Speed']);
						const period = parseFloat(g_params[wavePartOfName+'Period']);
						if (!isNaN(speed) && !isNaN(period)) {
							const wavelength = period / speed;
							const frequency = 1 / period;
							setParamAndHtmlFieldValue(wavePartOfName+'Wavelength', wavelength);
							setParamAndHtmlFieldValue(wavePartOfName+'Frequency', frequency);
						}
					}
					if (paramIsFrequency) {
						const speed = parseFloat(g_params[wavePartOfName+'Speed']);
						const frequency = parseFloat(g_params[wavePartOfName+'Frequency']);
						if (!isNaN(speed) && !isNaN(frequency)) {
							const period = 1 / frequency;
							const wavelength = period / speed;
							setParamAndHtmlFieldValue(wavePartOfName+'Period', period);
							setParamAndHtmlFieldValue(wavePartOfName+'Wavelength', wavelength);
						}
					}

					if (paramChangeShouldDebugDrawWave) {
						g_debugDrawSomeWave = g_debugDrawSomeWaveForNFrames;
						g_debugDrawDfIndex = iWave;
					}

					requestAnimationFrameIfNotAlreadyPlaying();
				}
			}

			// Event handling for input type=range is different in different browsers (and still really experimental)
			// Just moving the slider causes 'change' events in IE, 'input' events in Firefox, and both of them in Chrome! 
			// 'change' in Firefox fires when you release the mouse, but only if the new value is different from the original value
			input.addEventListener('input', storeAndReRender);
			input.addEventListener('change', storeAndReRender);
			// in firefox we get a 'mouseup' and then a 'change' immediately afterwards
			// since behavior is different in different browsers, there's no reliable way to
			// tell if that 'change' event signified the end of the drag, or the start of a new drag
			// input.addEventListener('mouseup', onEnd);
			// input.addEventListener('touchend', onEnd);
			// input.addEventListener('keyup', onEnd);
		}
	});



	function exportParametersToJSON() {
		const jsonTextarea = document.getElementById('jsonTextarea');
		jsonTextarea.value = JSON.stringify(g_params, null, 2);

		// #TODO: also put the parameters in the browser address bar,
		// so it's super easy to share with others
		// serializeConfigStateToUrl();
	}

	function importParametersFromJSON() {
		const jsonTextarea = document.getElementById('jsonTextarea').value;
		try {
			const parameters = JSON.parse(jsonTextarea);

			Object.keys(parameters).forEach(key => {
				let element = document.getElementById(key);
				if (!element) {
					element = document.querySelector(`input[type=radio][name="${key}"][value="${parameters[key]}"]`);
				}
				if (element) {
					if (element.type === 'checkbox') {
						element.checked = parameters[key];
					} else if (element.type === 'radio') {
						if (element) element.checked = true;
					} else {
						element.value = parameters[key];
					}

					// Trigger input event to update any dependent displays
					const event = new Event('input', { bubbles: true });
					element.dispatchEvent(event);
				}
				else {
					// there is no input corresponding to this parameter, so what could it be instead?
					if (g_params.hasOwnProperty(key) && typeof g_params[key] == typeof parameters[key]) {
						g_params[key] = parameters[key];
					}
				}
			});

			alert('Parameters successfully imported!');
		} catch (error) {
			alert('Invalid JSON: ' + error.message);
		}
	}

	document.getElementById('btnExportParametersToJSON').addEventListener('click', exportParametersToJSON);
	document.getElementById('btnImportParametersFromJSON').addEventListener('click', importParametersFromJSON);


	function getMousePosInCanvasFromEvent(canvas, clientX, clientY) {
		const mouseEventPos = (clientX.x !== undefined && clientY === undefined) ? clientX : vec2(clientX, clientY); // if you pass in a vec2 as single arg
		const rect = canvas.getBoundingClientRect();
		// console.log(mouseEventPos, rect);
		const canvasRectPos = vec2(rect.left, rect.top); // for some reason these are not always integers
		const canvasSize = vec2(canvas.width, canvas.height); // with a maximized window, this is the screen resolution, like 1980*1080 (Because I use setCanvasSizeToWindowSize. Otherwise it would be the <canvas width="100" height="100"> hardcoded in HTML)
		const canvasRectSize = vec2(rect.width, rect.height); // this might be smaller than the screen resolution due to Windows's scaling feature
		const scale = vec2Div(canvasSize, canvasRectSize); // this will derive the scaling configured in Windows settings, for example 1.25 or 1.5. is this the same as window.devicePixelRatio?
		return vec2Mul(vec2Difference(mouseEventPos, canvasRectPos), scale);
	};
	function printMousePosition(e) {
		// console.log(e.clientX, e.clientY);
		let mousePosInCanvas = getMousePosInCanvasFromEvent(e.target, e.clientX, e.clientY);
		// console.log(mousePosInCanvas);
		mousePosInCanvas.x = Math.round(mousePosInCanvas.x);
		mousePosInCanvas.y = Math.round(mousePosInCanvas.y);
		document.getElementById("mouseCursorPos").innerHTML = "X="+mousePosInCanvas.x+" Y="+mousePosInCanvas.y;
	};
	document.getElementById("glCanvas").addEventListener("mousemove", printMousePosition);


	// we've done a bunch of other stuff while the textures etc are downloading
	// but now it's time
	g_quasicrystalShader = await quasicrystalShaderPromise;
	// https://jameshfisher.com/2020/10/22/why-is-my-webgl-texture-upside-down/
	g_webglContext.pixelStorei(g_webglContext.UNPACK_FLIP_Y_WEBGL, true);
	g_tinyDummyTexture = createImageTexture(g_webglContext, await tinyDummyImagePromise)

	async function awaitMainImageDownloadsAndMakeTextures(mainImageDownloader, isFirstTime) {
		const mainImage = await mainImageDownloader.mainImagePromise;
		if (!mainImage)
			console.error('main image missing', mainImageDownloader.sourcePhotoFilenames.base);
		g_mainImageTexture = createImageTexture(g_webglContext, mainImage);
		g_distanceFieldTextures = [];
		// for (let i = 0; i < mainImageDownloader.distanceFieldPromises.length; i++)
		// 	g_distanceFieldTextures.push(createImageTexture(g_webglContext, await mainImageDownloader.distanceFieldPromises[i]));
		console.log(mainImage, g_mainImageTexture, g_distanceFieldTextures);

		if (!isFirstTime) {
			requestAnimationFrameNoDuplicate();
		}
	}
	await awaitMainImageDownloadsAndMakeTextures(mainImageDownloader, true);


	{
		function replaceDistanceFieldThumbnail(elementId, sourcePixels, sourceWidth, sourceHeight) {
			// draw a scaled-down version of this distance field onto a small canvas
			const thumbnailImg = document.getElementById(elementId);
			const thumbnailCanvas = document.createElement('canvas');
			thumbnailCanvas.width = thumbnailImg.width;
			thumbnailCanvas.height = thumbnailImg.height;
			thumbnailCanvas.style.backgroundColor = "#999999";
			thumbnailImg.replaceWith(thumbnailCanvas);
			thumbnailCanvas.id = thumbnailImg.id;
			const thumbnailCtx = thumbnailCanvas.getContext('2d');
			const thumbnailCanvasPixels = thumbnailCtx.getImageData(0,0, thumbnailCanvas.width,thumbnailCanvas.height);

			for (let dy = 0; dy < thumbnailCanvas.height; dy++) {
				for (let dx = 0; dx < thumbnailCanvas.width; dx++) {
					// sx = source x
					// dx = destination x
					const sx = Math.round(dx * sourceWidth / thumbnailCanvas.width);
					const sy = Math.round(dy * sourceHeight / thumbnailCanvas.height);
					const dindex = (dx + (dy * thumbnailCanvas.width)) * 4;
					const sindex = (sx + (sy * sourceWidth)) * 2;
					const signedDistance = sourcePixels[sindex] / 640;
					let color;
					if (signedDistance < 0) {
						color = LerpRgb([0,255,0], [255,255,255], Math.abs(signedDistance)*2);
					}
					else {
						color = LerpRgb([255,0,255], [255,255,255], Math.abs(signedDistance)*2);
					}
					color = LerpRgb([0,0,0], color, 0.2/Math.abs(sourcePixels[sindex+1] / 640));
					thumbnailCanvasPixels.data[dindex + 0] = color[0];
					thumbnailCanvasPixels.data[dindex + 1] = color[1];
					thumbnailCanvasPixels.data[dindex + 2] = color[2];
					thumbnailCanvasPixels.data[dindex + 3] = 255;
				}
			}

			thumbnailCtx.putImageData(thumbnailCanvasPixels, 0,0);
		}

		//g_webglContext.pixelStorei(g_webglContext.UNPACK_FLIP_Y_WEBGL, false);
		// for (let i = 0; i < 8; i++) {
		let i = 0;
		while (document.getElementById('wave'+i+'DistanceFieldThumbnail')) {
			const sourcePixels = createDistanceFieldSourcePixels(canvas.width, canvas.height, i);
			g_distanceFieldTextures[i] = createDistanceFieldTexture(g_webglContext, canvas.width, canvas.height, sourcePixels);
			replaceDistanceFieldThumbnail('wave'+i+'DistanceFieldThumbnail', sourcePixels, canvas.width, canvas.height);
			i++;
		}
		//g_webglContext.pixelStorei(g_webglContext.UNPACK_FLIP_Y_WEBGL, true);
	}


	// get uniform locations in shader and cache them, for performance
	function cacheUniformLocation(shader, uniformName) {
		if (shader.uniformLocations === undefined)
			shader.uniformLocations = {};
		// https://developer.mozilla.org/en-US/docs/Web/API/WebGLRenderingContext/getUniformLocation
		shader.uniformLocations[uniformName] = g_webglContext.getUniformLocation(shader, uniformName);
		if (shader.uniformLocations[uniformName] === null)
			console.warn(`getUniformLocation(${uniformName}) returned null. Maybe you just don't use it in your shader?`);
			// not neccessarily an error. if you have a uniform in your shader that you don't USE in your shader,
			// then getUniformLocation will return null for it. i guess the shader compilator has noticed unused uniforms
			// and removed them from the program.
	}
	cacheUniformLocation(g_quasicrystalShader, "timeSec");
	cacheUniformLocation(g_quasicrystalShader, "uMainTex");
	cacheUniformLocation(g_quasicrystalShader, "uNumDistanceFieldTextures");
	cacheUniformLocation(g_quasicrystalShader, "uDistanceFieldTex");
	cacheUniformLocation(g_quasicrystalShader, "uLineLength");
	cacheUniformLocation(g_quasicrystalShader, "uSpeed");
	cacheUniformLocation(g_quasicrystalShader, "uWavelength");
	cacheUniformLocation(g_quasicrystalShader, "uFrequencyFalloff");
	cacheUniformLocation(g_quasicrystalShader, "uAmplitude");
	cacheUniformLocation(g_quasicrystalShader, "uAmplitudeFalloff");
	cacheUniformLocation(g_quasicrystalShader, "uBlendOrAlternate");
	cacheUniformLocation(g_quasicrystalShader, "uAlternationWavelength");
	cacheUniformLocation(g_quasicrystalShader, "uShiftAlongStrength");
	cacheUniformLocation(g_quasicrystalShader, "uDebugDrawDfIndex");
	cacheUniformLocation(g_quasicrystalShader, "uOpacity");
	cacheUniformLocation(g_quasicrystalShader, "uOpacityTweak");
	cacheUniformLocation(g_quasicrystalShader, "uCenterOverlapBefore");
	cacheUniformLocation(g_quasicrystalShader, "uCenterOverlapAfter");
	cacheUniformLocation(g_quasicrystalShader, "uCurvature");
	cacheUniformLocation(g_quasicrystalShader, "uExperimental");
	// console.log(g_quasicrystalShader.uniformLocations);



	console.log('setup done');

	window.requestAnimationFrame(drawWebGl);
	// always do this _once_ even if g_isPlaying is false, so we draw a picture
	// but then if g_isPlaying is false we won't keep requesting more frames
}

function downloadImage(url, filename, isRequired = false) {
	const prom = new Promise((resolve, reject) => {
		if (!filename) {
			resolve(null);
		}
		else {
			url += filename;

			const image = new Image();
			image.onload = function() {
				resolve(image);
			};
			image.onerror = function() {
				if (isRequired)
					reject();
				else
					resolve(null);
			};
			image.src = url;
		}
	});
	return prom;
}

// can also be used for video, but will only grab the 1st frame in the video
// then you'll have to update the texture with a new video frame manually
function createImageTexture(gl, image, wrapping = gl.CLAMP_TO_EDGE) {
	if (!image)
		return null;
	
	const texture = gl.createTexture();
	gl.bindTexture(gl.TEXTURE_2D, texture);
	gl.texImage2D(gl.TEXTURE_2D,
		0, // level
		gl.RGBA, // internalFormat
		gl.RGBA, // srcFormat
		gl.UNSIGNED_BYTE, // srcType
		image);

	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, wrapping);
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, wrapping);
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

	return texture;
}

function createDistanceFieldSourcePixels(width, height, whichDistanceFunctionIndex) {
	const sourcePixels = new Float32Array(width * height * 2);
	for (let y = 0; y < height; y++) {
		for (let x = 0; x < width; x++) {
			const index = (x + (y * width)) * 2;
			let signedDistance = 0;
			const p = vec2(x, y);

			// aligned to the main edges in the image
			if (whichDistanceFunctionIndex == 0)
				signedDistance = sdfCircle(p, vec2(635, 117), 175);
			else if (whichDistanceFunctionIndex == 1)
				signedDistance = sdfLineSegment(p, vec2(550, 350), vec2(560, 710));
			else if (whichDistanceFunctionIndex == 2)
				signedDistance = sdfLineSegment(p, vec2(550, 350), vec2(911, 432));
			else if (whichDistanceFunctionIndex == 3)
				signedDistance = sdfLineSegment(p, vec2(0, 117), vec2(937, 423));
			else if (whichDistanceFunctionIndex == 4) {
				signedDistance = sdfLineSegment(p, vec2(937, 423), vec2(1280, 214));
				signedDistance[2] = 40000; // hack needed in shader with combineDistanceFields(). #TODO: fix properly
			}
			else if (whichDistanceFunctionIndex == 5)
				signedDistance = sdfLineSegment(p, vec2(937, 423), vec2(980, 720));
			else if (whichDistanceFunctionIndex == 6)
				signedDistance = sdfLineSegment(p, vec2(1152, 550), vec2(1280, 500));
			else if (whichDistanceFunctionIndex == 7)
				signedDistance = sdfLineSegment(p, vec2(1152, 550), vec2(1180, 720));
			/*
			if (whichDistanceFunctionIndex == 7)
				signedDistance = sdfCircle(p, vec2(635, 117), 180);
			else {
				// evenly distributed in 7 directions around the unit circle
				const angleRadians = whichDistanceFunctionIndex * Math.PI / 7;
				const normal = vec2(Math.cos(angleRadians), Math.sin(angleRadians));
				signedDistance = sdfLineFromNormal(p, vec2(width/2, height/2), normal);
			}
			*/
			console.assert(Array.isArray(signedDistance));
			sourcePixels[index+0] = signedDistance[0];
			sourcePixels[index+1] = signedDistance[1];
			g_lineLengths[whichDistanceFunctionIndex] = signedDistance[2];
		}
	}
	return sourcePixels;
}

// #TODO: more distance functions, like distance to ellipse
// https://iquilezles.org/articles/distfunctions2d/
function sdfCircle(p, center, radius) {
	const pa = vec2Difference(p, center);
	const distanceToEdge = (vec2Magnitude(pa) - radius);
	const angle = (Math.atan2(pa.y, pa.x) + 2 * Math.PI) % (2 * Math.PI);
	// we don't want the angle in degrees or radians
	// we want the actual distance in pixels traveled along the edge of the circle
	return [distanceToEdge, angle * radius, 2 * Math.PI * radius];
}

// a and b are the two endpoints of the line segment, as vec2 objects with x and y members
// returns an array with 3 distances
// result 0 the signed distance to the infinite line - the sign flips on either side of the line
// result 1 is the distance along the line segment, that is, at a 90 degree angle to result 0. distance 0 is at the point a, and negative before that
// result 2 is the length of the line segment, so this return value doesn't depend on the "p" parameter, only on a and b
// (previously, result 1 was the unsigned distance to the finite line segment, so the distance field formed a capsule - a rectangle plus two half circles as end caps)
function sdfLineSegment(p, a, b) {
	const pa = vec2Difference(p, a);
	const ba = vec2Difference(b, a);
	const hInf     = dot(pa, ba) / dot(ba, ba); // ratio along infinite line
	const hClamped = clamp(hInf, 0.0, 1.0); // ratio along line segment
	const pProjectedOntoLine        = vec2MulScalar(ba, hInf);
	const pProjectedOntoLineSegment = vec2MulScalar(ba, hClamped);
	const vecFromPToClosestPointOnLine        = vec2Difference(pProjectedOntoLine, pa);
	const vecFromPToClosestPointOnLineSegment = vec2Difference(pProjectedOntoLineSegment, pa);
	let absDistanceToLine        = vec2Magnitude(vecFromPToClosestPointOnLine);
	let absDistanceToLineSegment = vec2Magnitude(vecFromPToClosestPointOnLineSegment); // capsule shape
	const baRotated90 = vec2(-ba.y, ba.x);
	const sign = dot(baRotated90, vecFromPToClosestPointOnLine) < 0 ? -1 : 1;
	const lineSegmentLength = vec2Magnitude(ba);
	const distanceAlongLine = lineSegmentLength * hInf;
/*
	let relDistanceOutsideEndpoint = 0;
	if (hInf < 0)
		relDistanceOutsideEndpoint = -hInf;
	else if (hInf > 1)
		relDistanceOutsideEndpoint = hInf - 1;

	if (relDistanceOutsideEndpoint > 0) {
		//const something = absDistanceToLineSegment - absDistanceToLine;
		//absDistanceToLine /= (1 + something/50);
		// works, but results in very sharp bends that quickly become straight lines
		// i want a more gradual curve, so...
		const absDistanceOutsideEndpoint = relDistanceOutsideEndpoint * lineSegmentLength;
		const divisor = Lerp(210, 170, InverseLerp(1, 300, absDistanceToLine));
		const foo = Math.pow(absDistanceOutsideEndpoint/divisor, 2.0);
		absDistanceToLine /= (1+foo);
	}
*/
	return [absDistanceToLine * sign, distanceAlongLine, lineSegmentLength];
}

// a is a point on the line
// normal is the line's normal vector
function sdfLineFromNormal(p, a, normal) {
	const pa = vec2Difference(p, a);
	const h = distanceAlongRay(pa, normal);
	const pProjectedOntoNormal = vec2MulScalar(normal, h);
	const absDistance = vec2Magnitude(pProjectedOntoNormal);
	const sign = h < 0 ? -1 : 1;
	return absDistance * sign;
}

function createDistanceFieldTexture(gl, width, height, sourcePixels) {
	// console.log('createDistanceFieldTexture(', width, height, ')');
	// In WebGL 1, format must be the same as internalformat.
	// In WebGL 2, the allowed combinations are listed in this table:
	// https://registry.khronos.org/webgl/specs/latest/2.0/#TEXTURE_TYPES_FORMATS_FROM_DOM_ELEMENTS_TABLE
	// https://webgl2fundamentals.org/webgl/lessons/webgl-data-textures.html
	const texture = gl.createTexture();
	gl.bindTexture(gl.TEXTURE_2D, texture);
	gl.texImage2D(gl.TEXTURE_2D,
		0, // level
		gl.RG32F, // internalFormat
		width,
		height,
		0, // border
		gl.RG, // format
		gl.FLOAT, // srcType
		sourcePixels);

	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

	return texture;
}

async function loadTextFile(filename) {
	// console.log('loadTextFile('+filename+')');
	const sourcePromise = fetch(filename);
	const sourceResponse = await sourcePromise;
	if (!sourceResponse.ok)
		setErrorMessage(`Download ${filename} failed with status ${sourceResponse.status}`);
	const source = await sourceResponse.text();
	// console.log('end loadTextFile('+filename+')');
	return source;
}

async function loadShader(gl, vertFilename, fragFilename, logPolarTransformIncludePromise, kernelRingFunctionIncludePromise) {
	// console.log('loadShader('+vertFilename+', '+fragFilename+')');
	const vertexSourcePromise = fetch(vertFilename);
	const fragSourcePromise = fetch(fragFilename);

	const vertexSourceResponse = await vertexSourcePromise;
	if (!vertexSourceResponse.ok)
		setErrorMessage(`Download ${vertFilename} failed with status ${vertexSourceResponse.status}`);
	const vertexSource = await vertexSourceResponse.text();
	// console.log(vertexSource);
	const vertexShader = gl.createShader(gl.VERTEX_SHADER);
	gl.shaderSource(vertexShader, vertexSource);
	gl.compileShader(vertexShader);

	const fragSourceResponse = await fragSourcePromise;
	if (!fragSourceResponse.ok)
		setErrorMessage(`Download ${fragFilename} failed with status ${fragSourceResponse.status}`);
	let fragSource = await fragSourceResponse.text();
	// console.log(fragSource);
	if (fragSource.indexOf("#include-kernel-ring-function.frag") != -1) {
		// console.log('in loadShader('+vertFilename+', '+fragFilename+') await kernelRingFunctionIncludePromise');
		const includeFileText = await kernelRingFunctionIncludePromise;
		// console.log('in loadShader('+vertFilename+', '+fragFilename+') await kernelRingFunctionIncludePromise done');
		fragSource = fragSource.replace("#include-kernel-ring-function.frag", includeFileText);
		// console.log(fragSource);
	}
	if (fragSource.indexOf("#include-log-polar-transform.frag") != -1) {
		// console.log('in loadShader('+vertFilename+', '+fragFilename+') await logPolarTransformIncludePromise');
		const includeFileText = await logPolarTransformIncludePromise;
		// console.log('in loadShader('+vertFilename+', '+fragFilename+') await logPolarTransformIncludePromise done');
		fragSource = fragSource.replace("#include-log-polar-transform.frag", includeFileText);
		// console.log(fragSource);
	}
	const fragmentShader = gl.createShader(gl.FRAGMENT_SHADER);
	gl.shaderSource(fragmentShader, fragSource);
	gl.compileShader(fragmentShader);

	const program = gl.createProgram();
	gl.attachShader(program, vertexShader);
	gl.attachShader(program, fragmentShader);
	gl.linkProgram(program);
	gl.detachShader(program, vertexShader);
	gl.detachShader(program, fragmentShader);
	gl.deleteShader(vertexShader);
	gl.deleteShader(fragmentShader);
	if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
		const linkErrLog = gl.getProgramInfoLog(program);
		setErrorMessage(`Shader program ${vertFilename} ${fragFilename} did not link successfully. Error log: ${linkErrLog}`);
		return;
	}
	
	// console.log('end loadShader('+vertFilename+', '+fragFilename+')');
	return program;
}

function structureDfParamsForShader(numWaves) {
	const field = {
		speed: [],
		wavelength: [],
		frequencyFalloff: [],
		amplitude: [],
		amplitudeFalloff: [],
		blendOrAlternate: [],
		alternationWavelength: [],
		shiftAlongStrength: [],
		waveType: [],
	};

	// console.log(g_params);
	let i = 0;
	for (let iWave = 0; iWave < numWaves; iWave++) {
		console.assert(g_params["wave"+iWave+"Speed"]);
		if (g_params["wave"+iWave+"Speed"]) {
			field.speed[i] = parseFloat(g_params["wave"+iWave+"Speed"]);
			field.wavelength[i] = parseFloat(g_params["wave"+iWave+"Wavelength"]);
			field.frequencyFalloff[i] = parseFloat(g_params["wave"+iWave+"FrequencyFalloff"]);
			let isActive = false;
			if (g_params["wave"+iWave+"Active"])
				isActive = true;
			field.amplitude[i] = isActive ? parseFloat(g_params["wave"+iWave+"Amplitude"]) : 0;
			field.amplitudeFalloff[i] = parseFloat(g_params["wave"+iWave+"AmplitudeFalloff"]);
			field.blendOrAlternate[i] = parseFloat(g_params["wave"+iWave+"BlendOrAlternate"]);
			field.alternationWavelength[i] = parseFloat(g_params["wave"+iWave+"AlternationWavelength"]);
			field.shiftAlongStrength[i] = parseFloat(g_params["wave"+iWave+"ShiftAlongStrength"]);
			field.waveType[i] = parseInt(g_params["wave"+iWave+"Type"]);
		}
		else {
			// field.speed[i] = 1;
			// field.wavelength[i] = 20;
			// field.frequencyFalloff[i] = 0.5;
			// field.amplitude[i] = 1.0;
			// field.amplitudeFalloff[i] = 0.5;
			// field.waveType[i] = 0;
		}
		i++;
	}

	return field;
}

function requestAnimationFrameIfNotAlreadyPlaying() {
	if (!g_isPlaying)
		requestAnimationFrameNoDuplicate();
}
// if we don't do this dance, we'll accidentally queue up more frames than we intended, so for example g_debugDrawFrequencies will tick down faster
function requestAnimationFrameNoDuplicate() {
	if (g_animFrameId !== undefined)
		window.cancelAnimationFrame(g_animFrameId);
	g_animFrameId = window.requestAnimationFrame(drawWebGl);
}

function drawWebGl(timestamp) {
	draw(g_webglContext, timestamp);
}
function draw(gl, timestamp) {
	if (g_simulateSingleStep)
		console.log("simulate single step");

	// update fps timer once per second
	// (i first tried making an fps counter that updates every frame, but there was too much variation)
	if (g_isPlaying) {
		const timeAtDrawStart = performance.now();
		if (g_performanceTimerStart === undefined) {
			g_performanceTimerStart = timeAtDrawStart;
			g_numFramesThisSecond = 0;
		}
		const timeElapsedMs = timeAtDrawStart - g_performanceTimerStart;
		if (timeElapsedMs >= 1000) {
			document.getElementById('frameTimeMs').innerHTML = g_numFramesThisSecond + 'fps';
			g_performanceTimerStart = timeAtDrawStart;

			// note that, if there is a slowdown in the simulation, g_numFramesThisSecond can be really low, like 5 or such
			// but we don't want to use that as the basis for our animation speed calculations!
			// the timestep will be too large, might cause instability with our simple explicit forward euler integration
			// so instead, assume that the slowest fps on any device is 60, and adapt to faster devices, but not slower
			// also, we perform this estimation only once per second, and only take LARGE steps, ie we don't adapt to small variations like 59 to 61
			// because in my experience, the timing in web browsers isn't accurate enough to adapt properly
			// and the integration can become less stable if the timestep varies
			// so you end up causing more jitter, more harm than good
			const commonScreenRefreshRates = [60, 75, 90, 100, 120, 125, 144, 160, 240, 300, 360, 540];
			let smallestDifferenceYet = 10000;
			for (let i in commonScreenRefreshRates) {
				const difference = Math.abs(commonScreenRefreshRates[i] - g_numFramesThisSecond);
				if (smallestDifferenceYet > difference) {
					smallestDifferenceYet = difference;
					g_estimatedFps = commonScreenRefreshRates[i];
				}
			}

			g_numFramesThisSecond = 0;
		}
		g_numFramesThisSecond++;
		g_numFrames++;
	}


	// g_params contains everything, but not in the right format for passing into shaders
	// so convert it to the right format here before drawing each frame

	const fieldParamsForShader = structureDfParamsForShader(g_distanceFieldTextures.length);
	// console.log(fieldParamsForShader);


	{
		gl.useProgram(g_quasicrystalShader);

		gl.uniform1f(g_quasicrystalShader.uniformLocations["timeSec"], g_numFrames / 60); // #TODO: fix proper timing

		gl.activeTexture(gl.TEXTURE8);
		gl.bindTexture(gl.TEXTURE_2D, g_mainImageTexture);
		gl.uniform1i(g_quasicrystalShader.uniformLocations['uMainTex'], 8);

		const textureUnitIndices = [];
		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, g_distanceFieldTextures[0]);
		textureUnitIndices.push(0);
		// #TODO: loop instead of these repetitive if()s?
		if (g_distanceFieldTextures.length > 1) {
			gl.activeTexture(gl.TEXTURE1);
			gl.bindTexture(gl.TEXTURE_2D, g_distanceFieldTextures[1]);
			textureUnitIndices.push(1);
		}
		if (g_distanceFieldTextures.length > 2) {
			gl.activeTexture(gl.TEXTURE2);
			gl.bindTexture(gl.TEXTURE_2D, g_distanceFieldTextures[2]);
			textureUnitIndices.push(2);
		}
		if (g_distanceFieldTextures.length > 3) {
			gl.activeTexture(gl.TEXTURE3);
			gl.bindTexture(gl.TEXTURE_2D, g_distanceFieldTextures[3]);
			textureUnitIndices.push(3);
		}
		if (g_distanceFieldTextures.length > 4) {
			gl.activeTexture(gl.TEXTURE4);
			gl.bindTexture(gl.TEXTURE_2D, g_distanceFieldTextures[4]);
			textureUnitIndices.push(4);
		}
		if (g_distanceFieldTextures.length > 5) {
			gl.activeTexture(gl.TEXTURE5);
			gl.bindTexture(gl.TEXTURE_2D, g_distanceFieldTextures[5]);
			textureUnitIndices.push(5);
		}
		if (g_distanceFieldTextures.length > 6) {
			gl.activeTexture(gl.TEXTURE6);
			gl.bindTexture(gl.TEXTURE_2D, g_distanceFieldTextures[6]);
			textureUnitIndices.push(6);
		}
		if (g_distanceFieldTextures.length > 7) {
			gl.activeTexture(gl.TEXTURE7);
			gl.bindTexture(gl.TEXTURE_2D, g_distanceFieldTextures[7]);
			textureUnitIndices.push(7);
		}
		gl.uniform1iv(g_quasicrystalShader.uniformLocations['uDistanceFieldTex'], textureUnitIndices);
		gl.uniform1i(g_quasicrystalShader.uniformLocations['uNumDistanceFieldTextures'], g_distanceFieldTextures.length);

		gl.uniform1fv(g_quasicrystalShader.uniformLocations['uLineLength'], g_lineLengths);
		gl.uniform1fv(g_quasicrystalShader.uniformLocations['uSpeed'], fieldParamsForShader.speed);
		gl.uniform1fv(g_quasicrystalShader.uniformLocations['uWavelength'], fieldParamsForShader.wavelength);
		gl.uniform1fv(g_quasicrystalShader.uniformLocations['uFrequencyFalloff'], fieldParamsForShader.frequencyFalloff);
		gl.uniform1fv(g_quasicrystalShader.uniformLocations['uAmplitude'], fieldParamsForShader.amplitude);
		gl.uniform1fv(g_quasicrystalShader.uniformLocations['uAmplitudeFalloff'], fieldParamsForShader.amplitudeFalloff);
		gl.uniform1fv(g_quasicrystalShader.uniformLocations['uBlendOrAlternate'], fieldParamsForShader.blendOrAlternate);
		gl.uniform1fv(g_quasicrystalShader.uniformLocations['uAlternationWavelength'], fieldParamsForShader.alternationWavelength);
		gl.uniform1fv(g_quasicrystalShader.uniformLocations['uShiftAlongStrength'], fieldParamsForShader.shiftAlongStrength);
		{
			const dfi = (g_debugDrawSomeWave ? g_debugDrawDfIndex : -1);
			gl.uniform1i(g_quasicrystalShader.uniformLocations['uDebugDrawDfIndex'], dfi);
		}
		gl.uniform1f(g_quasicrystalShader.uniformLocations['uOpacity'], parseFloat(g_params.opacity));
		gl.uniform1f(g_quasicrystalShader.uniformLocations['uOpacityTweak'], parseFloat(g_params.opacityTweak));
		gl.uniform1f(g_quasicrystalShader.uniformLocations['uCenterOverlapBefore'], parseFloat(g_params.centerOverlapBefore));
		gl.uniform1f(g_quasicrystalShader.uniformLocations['uCenterOverlapAfter'], parseFloat(g_params.centerOverlapAfter));
		gl.uniform1f(g_quasicrystalShader.uniformLocations['uCurvature'], parseFloat(g_params.curvature));
		gl.uniform1f(g_quasicrystalShader.uniformLocations['uExperimental'], parseFloat(g_params.experimental));

		renderFullScreen(gl, g_quasicrystalShader);
	}
	// console.log(g_debugDrawSomeWave, g_debugDrawDfIndex);

	const shouldRequestAnimationFrameAgain = (g_isPlaying);

	if (g_debugDrawSomeWave) {
		if (shouldRequestAnimationFrameAgain)
			g_debugDrawSomeWave--;
		else // when paused, just draw it for 1 frame
			g_debugDrawSomeWave = 0;
	}

	if (g_simulateSingleStep)
		console.log("end single step");

	g_simulateSingleStep = false;
	if (shouldRequestAnimationFrameAgain)
		g_animFrameId = window.requestAnimationFrame(drawWebGl);
}

function renderFullScreen(gl, shader) {
	/*
	+--------------+
	|           __/|
	|        __/   |
	|     __/      |
	|  __/         |
	|_/            |
	+--------------+
	*/
	{
		let vertexPositionAttribute = gl.getAttribLocation(shader, "aPosition");
		var quad_vertex_buffer = gl.createBuffer();
		var quad_vertex_buffer_data = new Float32Array([ 
			-1.0, -1.0, 0.0,
			1.0, -1.0, 0.0,
			-1.0,  1.0, 0.0,
			-1.0,  1.0, 0.0,
			1.0, -1.0, 0.0,
			1.0,  1.0, 0.0]);
		gl.bindBuffer(gl.ARRAY_BUFFER, quad_vertex_buffer);
		gl.bufferData(gl.ARRAY_BUFFER, quad_vertex_buffer_data, gl.STATIC_DRAW);
		gl.vertexAttribPointer(vertexPositionAttribute, 3, gl.FLOAT, false, 0, 0);
		gl.enableVertexAttribArray(vertexPositionAttribute);
	}

	{
		let texCoordAttribute = gl.getAttribLocation(shader, "aTexCoord");
		var quad_vertex_texCoord_buffer = gl.createBuffer();
		var quad_vertex_texCoord_buffer_data = new Float32Array([ 
			0.0, 0.0,
			1.0, 0.0,
			0.0, 1.0,
			0.0, 1.0,
			1.0, 0.0,
			1.0, 1.0]);
		gl.bindBuffer(gl.ARRAY_BUFFER, quad_vertex_texCoord_buffer);
		gl.bufferData(gl.ARRAY_BUFFER, quad_vertex_texCoord_buffer_data, gl.STATIC_DRAW);
		gl.vertexAttribPointer(texCoordAttribute, 2, gl.FLOAT, false, 0, 0);
		gl.enableVertexAttribArray(texCoordAttribute);
	}

	// console.log("before drawArrays");
	gl.drawArrays(gl.TRIANGLES, 0, 6);
	// console.log("after drawArrays");
}

function clamp(val, min, max) {
	console.assert(!Number.isNaN(val), "!Number.isNaN(val)");
	return Math.max(Math.min(val, max), min);
}

function Lerp(from, to, t) {
	return (from * (1 - t)) + (to * t);
}

/**
* @brief The inverse of Lerp. t == InverseLerp(from, to, Lerp(from, to, t)) and vice versa
* When value = from, returns 0
* When value = to, returns 1
* When value is midway between from and to, returns 0.5
* When value is outside the range, returns <0 or >1 as you would expect
* Be careful, if to==from you get division by 0 and an undefined result
*/
function InverseLerp(from, to, value) {
	return (value - from) / (to - from);
}

// from and to are arrays of length 3
function LerpRgb(from, to, t) {
	return [
		Lerp(from[0], to[0], t),
		Lerp(from[1], to[1], t),
		Lerp(from[2], to[2], t)
	];
}

function vec2(x, y) {
	return {
		x: x,
		y: y
	};
}

function vec2Add(a, b) {
	return vec2(
		a.x + b.x,
		a.y + b.y
	);
}

function vec2Difference(a, b) {
	return {
		x: a.x - b.x,
		y: a.y - b.y
	};
}

function vec2Magnitude(a) {
	return Math.sqrt(vec2SqrMagnitude(a));
}

function vec2SqrMagnitude(a) {
	return a.x * a.x + a.y * a.y;
}

function vec2SqrDistance(a, b) {
	return vec2SqrMagnitude(vec2Difference(a, b));
}

function vec2MulScalar(vec, scalar) {
	return vec2(
		vec.x * scalar,
		vec.y * scalar
	);
}

function vec2Mul(a, b) {
	return vec2(
		a.x * b.x,
		a.y * b.y
	);
}

function vec2Equal(a, b) {
	return (a.x == b.x) && (a.y == b.y);
}

function vec2Normalize(vec) {
	return vec2MulScalar(vec, 1/Math.sqrt(vec2SqrMagnitude(vec)));
}

function vec2SetMagnitude(vec, newMagnitude) {
	return vec2MulScalar(vec, newMagnitude / Math.sqrt(vec2SqrMagnitude(vec)));
}

function vec2Div(a, b) {
	if (b.x === undefined) {
		return {
			x: a.x / b,
			y: a.y / b
		};
	}
	else {
		return {
			x: a.x / b.x,
			y: a.y / b.y
		};
	}
}

function vec2Negate(a) {
	return vec2(-a.x, -a.y);
}

function dot(a, b) {
	return a.x * b.x + a.y * b.y;
}

// http://www.softschools.com/math/pre_calculus/decomposing_a_vector_into_components/
// Vector decomposition is the general process of breaking one vector into two orthagonal vectors that, if added, add up to the original vector
// Vector u can now be written u = w1 + w2, where w1 is parallel to vector v and w1 is perpendicular/orthogonal to w2.
// The vector component w1 is also called the projection of vector u onto vector v, projv u.
// Note that v can be mirrored without affecting the result. Ie parallel_projection(u, v) == parallel_projection(u, -v)
function parallelProjection(u, v)
{
	console.assert(v.x != 0 || v.y != 0);
	return vec2MulScalar(v, dot(u, v) / vec2Magnitude(v));
}


// returns the distance from origo to (the projection of point u along ray v)
// if the dot is on the backwards extension of the ray, returns a value < 0
function distanceAlongRay(u, v)
{
	console.assert(v.x != 0 || v.y != 0);
	return dot(u, v) / vec2Magnitude(v);
}
