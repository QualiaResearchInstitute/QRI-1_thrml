// global variables have the g_ prefix
let g_animFrameId;
let g_simulateSingleStep = false;
let g_isPlaying = false;
let g_performanceTimerStart;
let g_numFramesThisSecond = 0;
let g_estimatedFps = 60; // will get estimated after 1 second of runtime

// web gl objects
let g_webglContext;
let g_mainShader;
let g_initOscillatorsShader;
let g_oscillationShader;
let g_drawKernelShader;
let g_scalingShader;
let g_oscillatorFramebuffer;

// assets
let g_mainImage; // #TODO: sourcePhoto or baseImage is a better term
let g_mainImageDepth;
let g_mainImageEdges;
let g_mainImageTexture;
let g_mainImageDepthTexture;
let g_mainImageEdgeTexture;
let g_displacementVideo;
let g_displacementVideoTexture;
let g_blendingPatternVideo;
let g_blendingPatternVideoTexture;
let g_deepDreamTextures = [];
let g_deepDreamVideo;
let g_deepDreamVideoTexture;

const MAX_CANVAS_WIDTH = 1600;
const MAX_CANVAS_HEIGHT = 1200;
const MIN_CANVAS_WIDTH = 640;

let g_customUploadState = createInitialCustomUploadState();
let g_applyCustomUploadHandler = null;
let g_customPreviewTargets = {
	base: null,
	edge: null,
	depth: null
};

if (typeof window !== 'undefined' && typeof window.DEBUG_CUSTOM_UPLOAD === 'undefined')
	window.DEBUG_CUSTOM_UPLOAD = false;

function debugCustomUpload(...args) {
	const debugEnabled = (typeof window !== 'undefined') ? window.DEBUG_CUSTOM_UPLOAD : false;
	if (debugEnabled)
		console.log('[custom-upload]', ...args);
}

function describeDrawable(source) {
	if (!source)
		return null;
	const descriptor = {};
	if (source.id)
		descriptor.id = source.id;
	if (source.tagName)
		descriptor.tagName = source.tagName;
	if (typeof source.width === 'number')
		descriptor.width = source.width;
	else if (typeof source.videoWidth === 'number')
		descriptor.width = source.videoWidth;
	if (typeof source.height === 'number')
		descriptor.height = source.height;
	else if (typeof source.videoHeight === 'number')
		descriptor.height = source.videoHeight;
	return descriptor;
}

let g_oscillatorFields = [
	makeField(),
	makeField(),
];
function makeField() {
	return {
		oscillatorTextures: [], // this is an array of 2 textures, so we can read from one and write to the other, then swap
		latestIndex: 0, // oscillatorTextures[latestIndex] is the one we most recently wrote to, so next time we should read from it
		oscillatorTextureWidth: 0,
	};
}

// Chimera domain system
let g_domains = []; // Array of domain objects
let g_domainMaskTexture = null; // WebGL texture encoding domain membership (R channel = domain index)
let g_domainBoundaryPaths = []; // Array of SVG path strings for visualization
let g_useAutoDomains = true; // Enable auto-domain detection from image
let g_autoDomainData = null; // Stores {imageData, numDomains, domainMap} from generateAutoDomains
let g_domainMaskCPUData = null; // CPU-side domain map {width, height, numDomains, data:Int32Array}
let g_domainOrderParameter = new Float32Array(8);
let g_domainChaosParameter = new Float32Array(8);
const DOMAIN_ORDER_SMOOTHING = 0.85;
const DOMAIN_COHERENCE_TEXTURE_WIDTH = 64;
const DOMAIN_COHERENCE_TEXTURE_HEIGHT = 64;
let g_domainCoherenceFramebuffer = null;
let g_domainCoherenceTexture = null;
let g_domainCoherenceReadBuffer = null;
let g_csdState = {
	active: false,
	startTime: 0,
	intensity: 0,
	depressionDuration: 8000,
	recoveryDuration: 12000
};
let g_lastCsdUpdateTime = null;

function createDomain(name, params) {
	// params should include: kernelRingCoupling, kernelRingDistances, kernelRingWidths, 
	// oscillatorFrequencyMin, oscillatorFrequencyMax, couplingToOtherField, etc.
	return {
		id: g_domains.length,
		name: name || `Domain ${g_domains.length}`,
		svgPath: '', // SVG path string defining the domain boundary
		params: params || {
			kernelRingCoupling: [0.0, 0.0, 0.0, 0.0],
			kernelRingDistances: [0.1, 0.3, 0.6, 0.9],
			kernelRingWidths: [0.05, 0.08, 0.12, 0.15],
			oscillatorFrequencyMin: 3.0,
			oscillatorFrequencyMax: 7.0,
			couplingToOtherField: 0.0,
			kernelMaxDistance: 5.0,
		}
	};
}

function addDomain(domain) {
	g_domains.push(domain);
	updateDomainMaskTexture();
}

function removeDomain(domainId) {
	g_domains = g_domains.filter(d => d.id !== domainId);
	// Reassign IDs
	g_domains.forEach((d, i) => d.id = i);
	updateDomainMaskTexture();
}

// Helper function to create example chimera domains
function createExampleChimeraDomains() {
	// Clear existing domains
	g_domains = [];
	
	// Domain 1: "Hyper-Stable Frame" - stable and synchronized
	const domain1 = createDomain("Hyper-Stable Frame", {
		kernelRingCoupling: [2.0, 1.5, -0.5, 0.0], // Strong positive coupling
		kernelRingDistances: [0.1, 0.3, 0.6, 0.9],
		kernelRingWidths: [0.05, 0.08, 0.12, 0.15],
		oscillatorFrequencyMin: 1.0,
		oscillatorFrequencyMax: 3.0,
		couplingToOtherField: 0.0,
		kernelMaxDistance: 5.0,
	});
	// SVG path for left half of canvas (normalized coordinates)
	domain1.svgPath = "M 0 0 L 0.5 0 L 0.5 1 L 0 1 Z";
	addDomain(domain1);
	
	// Domain 2: "The Volatile Engine" - chaotic and incoherent
	const domain2 = createDomain("The Volatile Engine", {
		kernelRingCoupling: [-1.0, 0.5, 1.5, -2.0], // Mixed coupling, more chaotic
		kernelRingDistances: [0.1, 0.3, 0.6, 0.9],
		kernelRingWidths: [0.08, 0.12, 0.18, 0.25], // Wider rings
		oscillatorFrequencyMin: 5.0,
		oscillatorFrequencyMax: 15.0, // Higher frequencies
		couplingToOtherField: 0.0,
		kernelMaxDistance: 8.0, // Larger kernel
	});
	// SVG path for right half of canvas
	domain2.svgPath = "M 0.5 0 L 1 0 L 1 1 L 0.5 1 Z";
	addDomain(domain2);
	
	console.log("Created example chimera domains:", g_domains);
}

// Parse SVG path string into array of points
function parseSVGPath(svgPath, width, height) {
	// Simple parser for basic SVG path commands (M, L, Z)
	// For production, consider using a proper SVG path parser library
	const points = [];
	const commands = svgPath.match(/[MLZ][^MLZ]*/g) || [];
	let currentX = 0, currentY = 0;
	
	for (const cmd of commands) {
		const type = cmd[0];
		const coords = cmd.slice(1).trim().split(/[\s,]+/).filter(s => s).map(parseFloat);
		
		if (type === 'M' || type === 'm') {
			// Move to
			if (coords.length >= 2) {
				currentX = type === 'M' ? coords[0] : currentX + coords[0];
				currentY = type === 'M' ? coords[1] : currentY + coords[1];
				points.push({x: currentX, y: currentY});
			}
		} else if (type === 'L' || type === 'l') {
			// Line to
			if (coords.length >= 2) {
				currentX = type === 'L' ? coords[0] : currentX + coords[0];
				currentY = type === 'L' ? coords[1] : currentY + coords[1];
				points.push({x: currentX, y: currentY});
			}
		} else if (type === 'Z' || type === 'z') {
			// Close path - connect back to first point
			if (points.length > 0) {
				points.push({x: points[0].x, y: points[0].y});
			}
		}
	}
	
	// Normalize coordinates to [0, 1] range if they appear to be in pixel coordinates
	if (points.length > 0) {
		const maxX = Math.max(...points.map(p => p.x));
		const maxY = Math.max(...points.map(p => p.y));
		if (maxX > 1 || maxY > 1) {
			// Assume pixel coordinates, normalize
			return points.map(p => ({x: p.x / width, y: p.y / height}));
		}
	}
	
	return points;
}

// Point-in-polygon test using ray casting algorithm
function pointInPolygon(x, y, polygon) {
	let inside = false;
	for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
		const xi = polygon[i].x, yi = polygon[i].y;
		const xj = polygon[j].x, yj = polygon[j].y;
		const intersect = ((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
		if (intersect) inside = !inside;
	}
	return inside;
}

// Create domain mask texture - encodes which domain each pixel belongs to
function updateDomainMaskTexture() {
	if (!g_webglContext) {
		return;
	}
	
	if (g_domainMaskTexture) {
		g_webglContext.deleteTexture(g_domainMaskTexture);
		g_domainMaskTexture = null;
	}
	g_domainMaskCPUData = null;
	
	const canvas = g_webglContext.canvas;
	const width = canvas.width;
	const height = canvas.height;
	
	const pixels = new Float32Array(width * height * 4);
	const domainMap = new Int32Array(width * height);
	let numDomains = 0;
	let maxDomainIndex = 0;
	
	if (g_useAutoDomains && g_autoDomainData && g_autoDomainData.domainMap && g_autoDomainData.numDomains > 0) {
		numDomains = Math.min(8, g_autoDomainData.numDomains);
		const sourceWidth = g_autoDomainData.imageData.width;
		const sourceHeight = g_autoDomainData.imageData.height;
		const sourceMap = g_autoDomainData.domainMap;
		const maxDomains = Math.max(1, numDomains - 1);
		for (let y = 0; y < height; y++) {
			for (let x = 0; x < width; x++) {
				const dstIndex = (x + y * width);
				const u = (x + 0.5) / width;
				const v = (y + 0.5) / height;
				const srcX = Math.min(sourceWidth - 1, Math.max(0, Math.floor(u * sourceWidth)));
				const srcY = Math.min(sourceHeight - 1, Math.max(0, Math.floor(v * sourceHeight)));
				const domainId = Math.min(numDomains - 1, Math.max(0, sourceMap[srcY * sourceWidth + srcX]));
				domainMap[dstIndex] = domainId;
				pixels[dstIndex * 4 + 0] = maxDomains > 0 ? domainId / maxDomains : 0.0;
				pixels[dstIndex * 4 + 1] = 0.0;
				pixels[dstIndex * 4 + 2] = 0.0;
				pixels[dstIndex * 4 + 3] = 1.0;
				maxDomainIndex = Math.max(maxDomainIndex, domainId);
			}
		}
	} else if (g_domains.length > 0) {
		numDomains = Math.min(8, g_domains.length);
		const domainPolygons = g_domains.map(domain => ({
			id: Math.min(numDomains - 1, domain.id),
			polygon: parseSVGPath(domain.svgPath, width, height)
		}));
		const maxDomains = Math.max(1, numDomains - 1);
		for (let y = 0; y < height; y++) {
			for (let x = 0; x < width; x++) {
				const dstIndex = (x + y * width);
				const u = x / width;
				const v = y / height;
				let domainId = 0;
				for (const domainPoly of domainPolygons) {
					if (domainPoly.polygon.length > 0 && pointInPolygon(u, v, domainPoly.polygon)) {
						domainId = domainPoly.id;
						break;
					}
				}
				domainId = Math.min(numDomains - 1, Math.max(0, domainId));
				domainMap[dstIndex] = domainId;
				pixels[dstIndex * 4 + 0] = maxDomains > 0 ? domainId / maxDomains : 0.0;
				pixels[dstIndex * 4 + 1] = 0.0;
				pixels[dstIndex * 4 + 2] = 0.0;
				pixels[dstIndex * 4 + 3] = 1.0;
				maxDomainIndex = Math.max(maxDomainIndex, domainId);
			}
		}
	} else {
		// No domains, nothing to do
		g_domainOrderParameter.fill(0.0);
		g_domainChaosParameter.fill(1.0);
		return;
	}
	
	numDomains = Math.min(8, Math.max(numDomains, maxDomainIndex + 1));
	
	g_domainMaskTexture = g_webglContext.createTexture();
	g_webglContext.bindTexture(g_webglContext.TEXTURE_2D, g_domainMaskTexture);
	g_webglContext.texImage2D(g_webglContext.TEXTURE_2D, 0,
		g_webglContext.RGBA32F, width, height, 0,
		g_webglContext.RGBA, g_webglContext.FLOAT, pixels);
	g_webglContext.texParameteri(g_webglContext.TEXTURE_2D, g_webglContext.TEXTURE_WRAP_S, g_webglContext.CLAMP_TO_EDGE);
	g_webglContext.texParameteri(g_webglContext.TEXTURE_2D, g_webglContext.TEXTURE_WRAP_T, g_webglContext.CLAMP_TO_EDGE);
	g_webglContext.texParameteri(g_webglContext.TEXTURE_2D, g_webglContext.TEXTURE_MIN_FILTER, g_webglContext.LINEAR);
	g_webglContext.texParameteri(g_webglContext.TEXTURE_2D, g_webglContext.TEXTURE_MAG_FILTER, g_webglContext.LINEAR);
	g_webglContext.bindTexture(g_webglContext.TEXTURE_2D, null);
	
	g_domainMaskCPUData = {
		width,
		height,
		numDomains,
		data: domainMap
	};
	
	// Reset cached order parameters for new domain configuration
	for (let i = 0; i < 8; i++) {
		if (i < numDomains) {
			g_domainOrderParameter[i] = 0.0;
			g_domainChaosParameter[i] = 1.0;
		} else {
			g_domainOrderParameter[i] = 0.0;
			g_domainChaosParameter[i] = 0.0;
		}
	}
}

function ensureDomainCoherenceResources(gl) {
	if (!g_domainCoherenceFramebuffer)
		g_domainCoherenceFramebuffer = gl.createFramebuffer();
	if (!g_domainCoherenceTexture) {
		g_domainCoherenceTexture = gl.createTexture();
		gl.bindTexture(gl.TEXTURE_2D, g_domainCoherenceTexture);
		gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F,
			DOMAIN_COHERENCE_TEXTURE_WIDTH, DOMAIN_COHERENCE_TEXTURE_HEIGHT, 0,
			gl.RGBA, gl.FLOAT, null);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
		gl.bindTexture(gl.TEXTURE_2D, null);
	}
}

function updateDomainCoherence(gl) {
	if (!g_domainMaskCPUData || g_domainMaskCPUData.numDomains <= 0)
		return;
	if (!g_scalingShader || g_scalingShader.uniformLocations === undefined)
		return;
	if (!g_oscillatorFields.length || !g_oscillatorFields[0].oscillatorTextures.length)
		return;

	const numDomains = Math.min(8, g_domainMaskCPUData.numDomains);
	if (numDomains <= 0)
		return;

	ensureDomainCoherenceResources(gl);

	const sampleWidth = DOMAIN_COHERENCE_TEXTURE_WIDTH;
	const sampleHeight = DOMAIN_COHERENCE_TEXTURE_HEIGHT;
	const oscField = g_oscillatorFields[0];
	const oscillatorTexture = oscField.oscillatorTextures[oscField.latestIndex];

	gl.useProgram(g_scalingShader);

	gl.activeTexture(gl.TEXTURE0);
	gl.bindTexture(gl.TEXTURE_2D, oscillatorTexture);
	gl.uniform1i(g_scalingShader.uniformLocations['uTexture'], 0);

	gl.bindFramebuffer(gl.FRAMEBUFFER, g_domainCoherenceFramebuffer);
	gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, g_domainCoherenceTexture, 0);

	gl.viewport(0, 0, sampleWidth, sampleHeight);
	renderFullScreen(gl, g_scalingShader);

	if (!g_domainCoherenceReadBuffer || g_domainCoherenceReadBuffer.length !== sampleWidth * sampleHeight * 4)
		g_domainCoherenceReadBuffer = new Float32Array(sampleWidth * sampleHeight * 4);

	gl.readPixels(0, 0, sampleWidth, sampleHeight, gl.RGBA, gl.FLOAT, g_domainCoherenceReadBuffer);

	gl.bindFramebuffer(gl.FRAMEBUFFER, null);
	gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);

	const maskWidth = g_domainMaskCPUData.width;
	const maskHeight = g_domainMaskCPUData.height;
	const maskData = g_domainMaskCPUData.data;

	const sumX = new Float32Array(numDomains);
	const sumY = new Float32Array(numDomains);
	const counts = new Uint32Array(numDomains);

	for (let y = 0; y < sampleHeight; y++) {
		for (let x = 0; x < sampleWidth; x++) {
			const sampleIndex = (x + y * sampleWidth) * 4;
			const cosVal = g_domainCoherenceReadBuffer[sampleIndex + 0];
			const sinVal = g_domainCoherenceReadBuffer[sampleIndex + 1];

			if (!Number.isFinite(cosVal) || !Number.isFinite(sinVal))
				continue;

			const u = (x + 0.5) / sampleWidth;
			const v = (y + 0.5) / sampleHeight;
			const maskX = Math.min(maskWidth - 1, Math.max(0, Math.floor(u * maskWidth)));
			const maskY = Math.min(maskHeight - 1, Math.max(0, Math.floor(v * maskHeight)));
			const domainId = Math.min(numDomains - 1, Math.max(0, maskData[maskY * maskWidth + maskX]));

			sumX[domainId] += cosVal;
			sumY[domainId] += sinVal;
			counts[domainId]++;
		}
	}

	for (let i = 0; i < numDomains; i++) {
		let R = g_domainOrderParameter[i];
		if (counts[i] > 0) {
			const avgX = sumX[i] / counts[i];
			const avgY = sumY[i] / counts[i];
			const magnitude = Math.sqrt(avgX * avgX + avgY * avgY);
			if (Number.isFinite(magnitude))
				R = magnitude;
		}
		const smoothed = DOMAIN_ORDER_SMOOTHING * g_domainOrderParameter[i] + (1.0 - DOMAIN_ORDER_SMOOTHING) * R;
		g_domainOrderParameter[i] = Math.min(1.0, Math.max(0.0, smoothed));
		g_domainChaosParameter[i] = Math.min(1.0, Math.max(0.0, 1.0 - g_domainOrderParameter[i]));
	}

	for (let i = numDomains; i < 8; i++) {
		g_domainOrderParameter[i] = 0.0;
		g_domainChaosParameter[i] = 0.0;
	}
}

function triggerCorticalDepression() {
	const now = performance.now ? performance.now() : Date.now();
	g_csdState.active = true;
	g_csdState.startTime = now;
	g_csdState.intensity = Math.max(g_csdState.intensity, 0.2);
	g_lastCsdUpdateTime = now;
	debugCustomUpload('csd:trigger', { now });
	requestAnimationFrameIfNotAlreadyPlaying();
}

function updateCorticalDepressionState(timestamp) {
	const now = (timestamp !== undefined) ? timestamp : (performance.now ? performance.now() : Date.now());
	if (g_lastCsdUpdateTime === null)
		g_lastCsdUpdateTime = now;
	const dt = now - g_lastCsdUpdateTime;
	g_lastCsdUpdateTime = now;

	let intensity = g_csdState.intensity;
	if (g_csdState.active) {
		const elapsed = now - g_csdState.startTime;
		const rise = g_csdState.depressionDuration;
		const fall = g_csdState.depressionDuration + g_csdState.recoveryDuration;
		if (elapsed <= rise) {
			const t = clamp(elapsed / Math.max(1, rise), 0.0, 1.0);
			intensity = Math.sin(t * 1.57079632679); // smooth rise to 1
		}
		else if (elapsed <= fall) {
			const t = clamp((elapsed - rise) / Math.max(1, g_csdState.recoveryDuration), 0.0, 1.0);
			intensity = Math.cos(t * 1.57079632679); // smooth fall back to 0
		}
		else {
			g_csdState.active = false;
			intensity = 0.0;
		}
	}
	else if (intensity > 0.0) {
		const decay = Math.max(1, g_csdState.recoveryDuration);
		intensity = Math.max(0.0, intensity - dt / decay);
	}

	g_csdState.intensity = clamp(intensity, 0.0, 1.0);
	if (g_csdState.intensity > 0.001 && !g_isPlaying)
		requestAnimationFrameIfNotAlreadyPlaying();
	return g_csdState.intensity;
}

// configuration, adjustable by the user
// many are passed directly into the shader as uniforms, but not all
let g_params = {
	photoIndex: 6,
	// this object will be filled with a lot more properties too, dynamically, based on the input fields in the html
};

let g_debugDrawFilterKernel = 0;
let g_debugDrawFilterKernelShrinkage = 0;
let g_debugDrawFilterKernelForField = 0; // stores the index of the field that we should debug draw for
const g_debugDrawFilterKernelForNFrames = 300;
let g_debugDrawFrequencies = 0;
let g_debugDrawFrequenciesForField = 0; // stores the index of the field that we should debug draw for
const g_debugDrawFrequenciesForNFrames = 120; // draw the oscillators' natural frequencies (instead of drawing the main simulation) for this many frames in a row
let g_frequencySlidersMinMax = []; // this is the fixed min="0.1" max="30" in html source, not the current values that our user selected

const sourcePhotos = [
	{
		base: "grass.jpg",
		edges: "grass-edges",
		numDeepDreamLayers: 1,
	},
	{
		base: "park.jpg",
		depth: "park-depth.png",
		edges: "park-edges.jpg",
		numDeepDreamLayers: 6,
	},
	{
		base: "redflowers.jpg",
		edges: "redflowers-edges.jpg",
		numDeepDreamLayers: 1,
	},
	{
		base: "dog.jpg",
		depth: "dog-depth.png",
		edges: "dog-edges.jpg",
		numDeepDreamLayers: 6,
	},
	{
		base: "pool.jpg",
		depth: "pool-depth.png",
		edges: "pool-edges.jpg",
		numDeepDreamLayers: 6,
	},
	{
		base: "rockwall.jpg",
		edges: "rockwall-edges.jpg",
		symmetry: "styles/sym_rockwall.jpg",
		numDeepDreamLayers: 6,
	},
	{
		base: "room.jpg",
		depth: "room-depth.png",
		edges: "room-edges.png",
		numDeepDreamLayers: 6,
	},
	{
		base: "sunnywall.jpg",
		edges: "sunnywall-edges.jpg",
		numDeepDreamLayers: 6,
	},
];


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
	// #TODO: Don't download shader.vert 6 times. Download it just once and reuse it.
	const kernelRingFunctionIncludePromise = loadTextFile('kernel-ring-function.frag');
	const logPolarTransformIncludePromise = loadTextFile('log-polar-transform.frag');
	const mainShaderPromise = loadShader(g_webglContext, 'shader.vert', 'shader.frag', logPolarTransformIncludePromise);
	const scalingShaderPromise = loadShader(g_webglContext, 'shader.vert', 'simple-passthru.frag');
	const initOscillatorsShaderPromise = loadShader(g_webglContext, 'shader.vert', 'init-oscillators.frag', logPolarTransformIncludePromise);
	const oscillationShaderPromise = loadShader(g_webglContext, 'shader.vert', 'oscillate.frag', logPolarTransformIncludePromise, kernelRingFunctionIncludePromise);
	const drawKernelShaderPromise = loadShader(g_webglContext, 'shader.vert', 'draw-kernel.frag', logPolarTransformIncludePromise, kernelRingFunctionIncludePromise);
	// Load images and videos from 'assets' folder
	const tinyDummyImagePromise = downloadImage('assets/', 'white1px.png', true);
	const g_deepDreamVideoPromise = downloadVideo('assets/snakey.mp4');
	const g_displacementVideoPromise = downloadVideo('assets/texturelooped60.mp4');
	const g_blendingPatternVideoPromise = downloadVideo('assets/texturelooped100_30.mp4');
	// do other stuff while waiting for the downloads


	function populateDeepDreamLayerSelect(numLayers) {
		const select = document.getElementById('deepDreamLayerIndex');
		if (!select)
			return;
		const count = Math.max(1, numLayers || 0);
		select.innerHTML = '';
		for (let i = 0; i < count; i++) {
			const option = document.createElement('option');
			option.value = i;
			option.textContent = 'Layer ' + i;
			select.appendChild(option);
		}
		if (g_params.deepDreamLayerIndex !== undefined && g_params.deepDreamLayerIndex >= count)
			g_params.deepDreamLayerIndex = count - 1;
	}

	function queueMainImageDownload(sourcePhotoFilenames) {
		const mainImageDownloader = {};
		mainImageDownloader.sourcePhotoFilenames = sourcePhotoFilenames;
		if (sourcePhotoFilenames.baseImageElement)
			mainImageDownloader.mainImagePromise = Promise.resolve(sourcePhotoFilenames.baseImageElement);
		else
			mainImageDownloader.mainImagePromise = downloadImage('assets/', sourcePhotoFilenames.base, true);
		if (sourcePhotoFilenames.depthImageElement !== undefined)
			mainImageDownloader.mainImageDepthPromise = Promise.resolve(sourcePhotoFilenames.depthImageElement);
		else
			mainImageDownloader.mainImageDepthPromise = downloadImage('assets/', sourcePhotoFilenames.depth);
		if (sourcePhotoFilenames.edgeImageElement !== undefined)
			mainImageDownloader.mainImageEdgesPromise = Promise.resolve(sourcePhotoFilenames.edgeImageElement);
		else
			mainImageDownloader.mainImageEdgesPromise = downloadImage('assets/', sourcePhotoFilenames.edges);
		mainImageDownloader.deepDreamPromises = [];
		if (sourcePhotoFilenames.deepDreamLayerElements && sourcePhotoFilenames.deepDreamLayerElements.length) {
			for (let i = 0; i < sourcePhotoFilenames.deepDreamLayerElements.length; i++)
				mainImageDownloader.deepDreamPromises.push(Promise.resolve(sourcePhotoFilenames.deepDreamLayerElements[i]));
		}
		else {
			for (let i = 0; i < sourcePhotoFilenames.numDeepDreamLayers; i++) {
				let filename = sourcePhotoFilenames.base;
				filename = filename.substr(0, filename.lastIndexOf(".")) + "_" + (i+1) + ".jpg";
				mainImageDownloader.deepDreamPromises.push(downloadImage('assets/deep_dream_layers/', filename));
			}
		}
		populateDeepDreamLayerSelect(sourcePhotoFilenames.numDeepDreamLayers);
		return mainImageDownloader;
	}

	for (let i = 0; i < sourcePhotos.length; i++) {
		const thumbnailWidth = 100;
		const aspectRatio = 0.5625;
		const thumbnailHeight = Math.round(thumbnailWidth * aspectRatio);
		// document.getElementById("sourcePhotos").innerHTML += '<img src="assets/' + sourcePhotos[i].base + '" width="' + thumbnailWidth + '" height="' + thumbnailHeight + '">';
		const img = document.createElement("img");
		img.src = 'assets/' + sourcePhotos[i].base;
		img.width = thumbnailWidth;
		img.height = thumbnailHeight;
		img.addEventListener('click', function() {
			g_params.photoIndex = i;
			// dynamically load in only the textures that change. keep the rest
			const mainImageDownloader = queueMainImageDownload(sourcePhotos[i]);
			awaitMainImageDownloadsAndMakeTextures(mainImageDownloader, false);
		});
		document.getElementById("sourcePhotos").appendChild(img);
	}

	// fire off a bunch of downloads as Promises, in parallel
	// Load images and videos from 'assets' folder
	const mainImageDownloader = queueMainImageDownload(sourcePhotos[g_params.photoIndex]);
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
		g_displacementVideo.pause();
		g_blendingPatternVideo.pause();
		g_deepDreamVideo.pause();
	});
	document.getElementById('btnPlay').addEventListener('click', (e) => {
		if (!g_isPlaying) {
			g_isPlaying = true;
			updatePlayPauseButtons();
			g_displacementVideo.play();
			g_blendingPatternVideo.play();
			g_deepDreamVideo.play();
			g_performanceTimerStart = performance.now();
			g_numFramesThisSecond = 0;
			requestAnimationFrameNoDuplicate();
		}
	});
	document.getElementById('btnSingleStep').addEventListener('click', (e) => {
		if (!g_isPlaying) {
			g_simulateSingleStep = true;
			g_displacementVideo.currentTime    = (g_displacementVideo.currentTime    + 1/g_estimatedFps) % g_displacementVideo.duration;
			g_blendingPatternVideo.currentTime = (g_blendingPatternVideo.currentTime + 1/g_estimatedFps) % g_blendingPatternVideo.duration;
			g_deepDreamVideo.currentTime       = (g_deepDreamVideo.currentTime       + 1/g_estimatedFps) % g_deepDreamVideo.duration;
			requestAnimationFrameNoDuplicate();
		}
	});
	for (let iField in g_oscillatorFields) {
		document.getElementById('layer'+iField+'btnRandomizePhases').addEventListener('click', (e) => {
			randomizeOscillatorPhases(g_webglContext, iField);
		});
		document.getElementById('layer'+iField+'btnUnifyPhases').addEventListener('click', (e) => {
			unifyOscillatorPhases(g_webglContext, iField);
		});
		document.getElementById('layer'+iField+'btnPlaneWavePhases').addEventListener('click', (e) => {
			resetOscillatorPhasesToPlaneWaves(g_webglContext, iField, 0);
		});
		document.getElementById('layer'+iField+'btnPlaneWaveVPhases').addEventListener('click', (e) => {
			resetOscillatorPhasesToPlaneWaves(g_webglContext, iField, Math.PI * 1.5);
		});
		document.getElementById('layer'+iField+'btnPlaneWave45Phases').addEventListener('click', (e) => {
			resetOscillatorPhasesToPlaneWaves(g_webglContext, iField, Math.PI * (1.25 + iField * 0.5));
		});
	}
	document.getElementById('btnRandomizeAllPhases').addEventListener('click', () => {
		randomizeAllPhases(g_webglContext);
	});
	document.getElementById('btnUnifyAllPhases').addEventListener('click', () => {
		unifyAllPhases(g_webglContext);
	});
	const triggerCsdButton = document.getElementById('btnTriggerCorticalDepression');
	if (triggerCsdButton)
		triggerCsdButton.addEventListener('click', triggerCorticalDepression);
	initCustomUploadControls();
	g_applyCustomUploadHandler = async function() {
		const descriptor = buildCustomSourceDescriptor();
		if (!descriptor)
			return;
		const mainImageDownloader = queueMainImageDownload(descriptor);
		g_params.photoIndex = -1;
		await awaitMainImageDownloadsAndMakeTextures(mainImageDownloader, false);
	};

	// both layers should have the same .min and .max for these sliders
	{
		const oscillatorFrequencyMinSlider = document.getElementById('layer0oscillatorFrequencyMin');
		const oscillatorFrequencyMaxSlider = document.getElementById('layer0oscillatorFrequencyMax');
		g_frequencySlidersMinMax = [
			parseFloat(oscillatorFrequencyMinSlider.min),
			parseFloat(oscillatorFrequencyMaxSlider.max)
		];
	}


	function printKernelEffectiveRadius(iField) {
		let [scaleFactor, scaledKernelMaxDistance, oscillatorTextureWidth, oscillatorTextureHeight] = calculateScalingForPerformance(iField);
		const kernelShrinkFactor = parseFloat(g_params["layer"+iField+"kernelShrinkFactor"]);
		let kernelExplanationString = g_params["layer"+iField+"kernelMaxDistance"] + "/" + scaleFactor + " = " + scaledKernelMaxDistance;
		if (kernelShrinkFactor != 1.0 && g_params["layer"+iField+"kernelSizeShrunkBy"] != "nothing")
			kernelExplanationString += " down to " + g_params["layer"+iField+"kernelMaxDistance"] + "*" + kernelShrinkFactor + "/" + scaleFactor + " = " + (scaledKernelMaxDistance * kernelShrinkFactor).toFixed(2);
		document.getElementById("kernelEffectiveRadius"+iField).innerHTML = kernelExplanationString;
		document.getElementById("fieldEffectiveResolution"+iField).innerHTML = oscillatorTextureWidth + " * " + oscillatorTextureHeight;
	}


	function isNumActiveLayersGreaterThan(iField) {
		return parseInt(g_params.numOscillatorFields) > iField;
	}

	// any change to input form parameters means:
	// we should store the new value in the g_params object (for faster lookup during rendering, and easy exporting to JSON)
	// we should re-render
	// beyond that, it's different for different parameters
	let inputEventTriggeredDirectlyByUser = true;
	const inputs = document.querySelectorAll('input, select');
	inputs.forEach(input => {
		if ((input.id && input.id != "paramPreset") || (input.name && input.type == 'radio')) {

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
			let paramChangeShouldDrawKernel = false;
			let paramChangeShouldDrawKernelShrinkage = false;
			let paramChangeShouldDrawFrequencies = false;
			let paramAffectsOscillatorTexture = false;
			let paramAffectsKernelEffectiveRadius = false;
			let paramDisEnablesFrequencyCorrelationPatternSize = false;
			let iField = -1;
			let restOfName;
			{
				const layerRegexp = /^layer([0-9])/g;
				const matches = layerRegexp.exec(paramName);
				if (matches) {
					iField = parseInt(matches[1]);
					restOfName = paramName.substr(layerRegexp.lastIndex);
					// console.log(paramName, iField, layerRegexp.lastIndex, restOfName);
					if (restOfName == 'kernelShrinkFactor'
					|| restOfName == 'kernelSizeShrunkBy') {
						paramAffectsOscillatorTexture = true;
						paramAffectsKernelEffectiveRadius = true;
						paramChangeShouldDrawKernelShrinkage = true;
					}
					else if (restOfName == 'logPolar') {
						paramAffectsOscillatorTexture = true;
					}
					else if (restOfName == 'oscillatorFrequencyMax'
					|| restOfName == 'oscillatorFrequencyMin'
					|| restOfName == 'frequenciesVaryBy'
					|| restOfName == 'frequencyCorrelationPatternSize') {
						paramAffectsOscillatorTexture = true;
						paramChangeShouldDrawFrequencies = true;
					}
					else if (restOfName == 'fieldResolutionHalvings'
					|| restOfName == 'kernelMaxDistance') {
						paramAffectsKernelEffectiveRadius = true;
						paramChangeShouldDrawKernel = true;
					}
					else if (/^ring[0-9]coupling$/.test(restOfName)
					|| /^ring[0-9]distance$/.test(restOfName)
					|| /^ring[0-9]width$/.test(restOfName)) {
						paramChangeShouldDrawKernel = true;
					}

					// conditions for when layer inputs should be enabled / disabled
					// .enabledCondition is not a thing that already exists on DOM objects, we just invent it and add it on our own
					// the enabledCondition function will be called later
					if (restOfName == 'kernelShrinkFactor')
						input.enabledCondition = function() {
							return (g_params['layer'+iField+'kernelSizeShrunkBy'] != 'nothing') && isNumActiveLayersGreaterThan(iField);
						};
					else if (restOfName == 'frequencyCorrelationPatternSize')
						input.enabledCondition = function() {
							return (g_params['layer'+iField+'frequenciesVaryBy'] == 'pattern') && isNumActiveLayersGreaterThan(iField);
						};
					else
						input.enabledCondition = function() { return isNumActiveLayersGreaterThan(iField); };
				}
			}

			// conditions for when inputs should be enabled / disabled
			if (paramName == 'customVideoPlaybackSpeed')
				input.enabledCondition = function() {
					return g_params.useCustomVideo;
				};
			else if (paramName == 'crossFieldBlendingBalance')
				input.enabledCondition = function() { return isNumActiveLayersGreaterThan(1); };
			else if (paramName == 'layer0influenceFromOtherLayer')
				input.enabledCondition = function() { return isNumActiveLayersGreaterThan(1); };

			// console.log(input.type, input.id, input.name, paramChangeShouldDrawKernel, paramChangeShouldDrawFrequencies, paramAffectsOscillatorTexture, paramAffectsKernelEffectiveRadius);

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

			function setVideoSpeed() {
				if (input.id == 'driftPatternPlaybackSpeed') g_displacementVideo.playbackRate = input.value;
				if (input.id == 'blendPatternPlaybackSpeed') g_blendingPatternVideo.playbackRate = input.value;
				if (input.id == 'customVideoPlaybackSpeed') g_deepDreamVideo.playbackRate = input.value;
			}

			// first do this once, at startup:
			storeHtmlInputValueInGlobalParamsObject();
			// setVideoSpeed(); // #TODO: fix TypeError: g_displacementVideo is undefined. because we haven't created it yet. it happens later.
			// then add event listeners to do it every time the user changes the param

			function storeAndReRender() {
				// console.log("storeAndReRender", input);
				if (storeHtmlInputValueInGlobalParamsObject()) {

					if (paramAffectsKernelEffectiveRadius)
						printKernelEffectiveRadius(iField);

					if (paramAffectsOscillatorTexture)
						changeFrequenciesAndKernelShrinkageButKeepPhases(g_webglContext, iField);
						// #TODO: if user toggled logPolar checkbox, would be nice to resample the phases to the new coordinate system, like we already do with couplingToOtherField in oscillate.frag

					if (inputEventTriggeredDirectlyByUser) { // don't debug draw anything when importing a whole set of parameters from preset / url / json
						if (paramChangeShouldDrawKernel) {
							g_debugDrawFilterKernel = g_debugDrawFilterKernelForNFrames;
							g_debugDrawFilterKernelForField = iField;
						}

						if (paramChangeShouldDrawKernelShrinkage) {
							g_debugDrawFilterKernelShrinkage = g_debugDrawFrequenciesForNFrames;
							g_debugDrawFilterKernelForField = iField;
						}

						if (paramChangeShouldDrawFrequencies) {
							g_debugDrawFrequencies = g_debugDrawFrequenciesForNFrames;
							g_debugDrawFrequenciesForField = iField;
						}
					}

					setVideoSpeed();
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

	for (let iField = 0; iField < g_oscillatorFields.length; iField++)
		printKernelEffectiveRadius(iField);



	function importParametersFromJsonString(jsonTextarea) {
		try {
			const parameters = JSON.parse(jsonTextarea);
			importParametersFromJsonObj(parameters);
			return true;
		}
		catch (error) {
			alert('Invalid JSON: ' + error.message);
			return false;
		}
	}

	// query string as submitted by a classic html <form>, formatted like ?foo=344&bar=false&baz=rainbow
	function importParametersFromUrlQueryString(queryString) {
		const parameters = objectFromUrlQueryString(queryString);
		importParametersFromJsonObj(parameters);
	}

	function importParametersFromJsonObj(parameters) {
		inputEventTriggeredDirectlyByUser = false;
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
				// but note that we have set inputEventTriggeredDirectlyByUser=false
				const event = new Event('input', { bubbles: true });
				element.dispatchEvent(event);
			}
			else {
				// there is no input corresponding to this parameter, so what could it be instead?
				if (g_params.hasOwnProperty(key) && typeof g_params[key] == typeof parameters[key]) {
					g_params[key] = parameters[key];
					if (key == 'photoIndex') {
						const mainImageDownloader = queueMainImageDownload(sourcePhotos[parameters[key]]);
						awaitMainImageDownloadsAndMakeTextures(mainImageDownloader, false);
					}
				}
			}
		});
		inputEventTriggeredDirectlyByUser = true;
	}

	document.getElementById('btnImportParametersFromJson').addEventListener('click', function (e) {
		const jsonTextarea = document.getElementById('jsonImportTextarea').value;
		if (importParametersFromJsonString(jsonTextarea))
			alert('Parameters successfully imported!');
	});
	document.getElementById('exportParamsModal').addEventListener('show.bs.modal', function (e) {
		const sharingUrlText = document.querySelector('#exportParamsModal [name="sharingUrl"]');
		sharingUrlText.value = window.location.origin + window.location.pathname + '?' + encodeURI(urlQueryStringFromObject(g_params));
		const jsonTextarea = document.querySelector('#jsonExportTextarea');
		jsonTextarea.value = JSON.stringify(g_params, null, 2);
	});

	{
		const paramPresets = [
			{
				internalNameInStudy: "dog_bw_lp1_kr5_rc-505",
				url: "driftMaxDisplacement=0&driftPatternPlaybackSpeed=0.2&driftPatternSize=1280&deepDreamLayerIndex=0&useCustomVideo=false&customVideoPlaybackSpeed=0.5&deepDreamOpacity=0&deepLuminosityBlendOpacity=0.5&blendPatternPlaybackSpeed=0.2&blendPatternSize=1280&oscillatorOpacity=0.25&oscillatorColors=black-white-soft-normal&numOscillatorFields=1&crossFieldBlendingBalance=1.5&layer0logPolar=true&layer0fieldResolutionHalvings=0&layer0kernelMaxDistance=5&layer0kernelSizeShrunkBy=nothing&layer0kernelShrinkFactor=0.3&layer0ringcoupling=-5_0_5_0&layer0ringdistance=0.1_0.3_0.6_0.9&layer0ringwidth=0.05_0.08_0.12_0.15&layer0influenceFromOtherLayer=0&layer0oscillatorFrequency=0.5_0.5&layer0frequencyCorrelationPatternSize=100&layer1logPolar=false&layer1fieldResolutionHalvings=1&layer1kernelMaxDistance=5&layer1kernelSizeShrunkBy=nothing&layer1kernelShrinkFactor=0.3&layer1ringcoupling=0_0_0_0&layer1ringdistance=0.1_0.3_0.6_0.9&layer1ringwidth=0.05_0.08_0.12_0.15&layer1influenceFromOtherLayer=0&layer1oscillatorFrequency=3_7&layer1frequencyCorrelationPatternSize=100"
			},
			{
				internalNameInStudy: "02_smooth_clean_2cb",
				url: "driftMaxDisplacement=0&driftPatternPlaybackSpeed=0.2&driftPatternSize=1280&deepDreamLayerIndex=0&useCustomVideo=false&customVideoPlaybackSpeed=0.5&deepDreamOpacity=0&deepLuminosityBlendOpacity=0.5&blendPatternPlaybackSpeed=0.2&blendPatternSize=1280&oscillatorOpacity=0.6&oscillatorColors=rainbow&numOscillatorFields=2&crossFieldBlendingBalance=1.5&layer0logPolar=true&layer0fieldResolutionHalvings=1&layer0kernelMaxDistance=47&layer0kernelSizeShrunkBy=edge&layer0kernelShrinkFactor=0.3&layer0ringcoupling=2.5_3_4.5_4.5&layer0ringdistance=0.1_0.3_0.4_0.9&layer0ringwidth=0.05_0.08_0.12_0.15&layer0influenceFromOtherLayer=3&layer0oscillatorFrequency=0.6_2.6&layer0frequencyCorrelationPatternSize=100&layer1logPolar=true&layer1fieldResolutionHalvings=3&layer1kernelMaxDistance=5&layer1kernelSizeShrunkBy=nothing&layer1kernelShrinkFactor=0.3&layer1ringcoupling=-4.5_3.5_-2.5_0&layer1ringdistance=0.1_0.3_0.6_0.9&layer1ringwidth=0.05_0.08_0.12_0.15&layer1influenceFromOtherLayer=3&layer1oscillatorFrequency=0.4_2.9&layer1frequencyCorrelationPatternSize=100"
			},
			{
				internalNameInStudy: "03_complex_multi_layer_checkerboard_dmt",
				url: "driftMaxDisplacement=0&driftPatternPlaybackSpeed=0.2&driftPatternSize=1280&deepDreamLayerIndex=0&useCustomVideo=false&customVideoPlaybackSpeed=0.5&deepDreamOpacity=0&deepLuminosityBlendOpacity=0.5&blendPatternPlaybackSpeed=0.2&blendPatternSize=1280&oscillatorOpacity=0.6&oscillatorColors=rainbow&numOscillatorFields=2&crossFieldBlendingBalance=1.5&layer0logPolar=false&layer0fieldResolutionHalvings=1&layer0kernelMaxDistance=47&layer0kernelSizeShrunkBy=edge&layer0kernelShrinkFactor=0.3&layer0ringcoupling=2.5_3_4.5_4.5&layer0ringdistance=0.1_0.3_0.4_0.9&layer0ringwidth=0.05_0.08_0.12_0.15&layer0influenceFromOtherLayer=-0.5&layer0oscillatorFrequency=0.6_2.6&layer0frequencyCorrelationPatternSize=100&layer1logPolar=true&layer1fieldResolutionHalvings=3&layer1kernelMaxDistance=5&layer1kernelSizeShrunkBy=nothing&layer1kernelShrinkFactor=0.3&layer1ringcoupling=4_-2_-1_0&layer1ringdistance=0.1_0.3_0.6_0.9&layer1ringwidth=0.05_0.08_0.12_0.15&layer1influenceFromOtherLayer=3&layer1oscillatorFrequency=0.4_2.9&layer1frequencyCorrelationPatternSize=100"
			},
			{
				internalNameInStudy: "01_seizure_2cb",
				url: "driftMaxDisplacement=0&driftPatternPlaybackSpeed=0.2&driftPatternSize=1280&deepDreamLayerIndex=0&useCustomVideo=false&customVideoPlaybackSpeed=0.5&deepDreamOpacity=0&deepLuminosityBlendOpacity=0.5&blendPatternPlaybackSpeed=0.2&blendPatternSize=1280&oscillatorOpacity=0.6&oscillatorColors=rainbow&numOscillatorFields=2&crossFieldBlendingBalance=1.5&layer0logPolar=true&layer0fieldResolutionHalvings=1&layer0kernelMaxDistance=47&layer0kernelSizeShrunkBy=edge&layer0kernelShrinkFactor=0.3&layer0ringcoupling=2.5_3_4.5_4.5&layer0ringdistance=0.1_0.3_0.4_0.9&layer0ringwidth=0.05_0.08_0.12_0.15&layer0influenceFromOtherLayer=3&layer0oscillatorFrequency=0.6_2.6&layer0frequencyCorrelationPatternSize=100&layer1logPolar=true&layer1fieldResolutionHalvings=3&layer1kernelMaxDistance=5&layer1kernelSizeShrunkBy=nothing&layer1kernelShrinkFactor=0.3&layer1ringcoupling=-4.5_3.5_4.5_-5&layer1ringdistance=0.1_0.3_0.6_0.9&layer1ringwidth=0.05_0.08_0.12_0.15&layer1influenceFromOtherLayer=3&layer1oscillatorFrequency=0.4_2.9&layer1frequencyCorrelationPatternSize=100"
			},
			{
				internalNameInStudy: "09_several_phases_ghb_dmt",
				url: "driftMaxDisplacement=0&driftPatternPlaybackSpeed=0.2&driftPatternSize=1280&deepDreamLayerIndex=0&useCustomVideo=false&customVideoPlaybackSpeed=0.5&deepDreamOpacity=0&deepLuminosityBlendOpacity=0.5&blendPatternPlaybackSpeed=0.2&blendPatternSize=1280&oscillatorOpacity=0.6&oscillatorColors=rainbow2x&numOscillatorFields=2&crossFieldBlendingBalance=1.5&layer0logPolar=true&layer0fieldResolutionHalvings=1&layer0kernelMaxDistance=47&layer0kernelSizeShrunkBy=edge&layer0kernelShrinkFactor=0.3&layer0ringcoupling=2.5_4_-3_0.5&layer0ringdistance=0.1_0.3_0.4_0.9&layer0ringwidth=0.05_0.08_0.12_0.15&layer0influenceFromOtherLayer=-2&layer0oscillatorFrequency=0.6_2.6&layer0frequencyCorrelationPatternSize=100&layer1logPolar=false&layer1fieldResolutionHalvings=1&layer1kernelMaxDistance=46&layer1kernelSizeShrunkBy=nothing&layer1kernelShrinkFactor=0.3&layer1ringcoupling=-4_-0.5_1_-1&layer1ringdistance=0.1_0.3_0.6_0.9&layer1ringwidth=0.05_0.08_0.12_0.15&layer1influenceFromOtherLayer=0.5&layer1oscillatorFrequency=0.4_2.9&layer1frequencyCorrelationPatternSize=100"
			},
			{
				internalNameInStudy: "05_creamy_dmt_lsd",
				url: "driftMaxDisplacement=0&driftPatternPlaybackSpeed=0.2&driftPatternSize=1280&deepDreamLayerIndex=0&useCustomVideo=false&customVideoPlaybackSpeed=0.5&deepDreamOpacity=0&deepLuminosityBlendOpacity=0.5&blendPatternPlaybackSpeed=0.2&blendPatternSize=1280&oscillatorOpacity=0.6&oscillatorColors=rainbow2x&numOscillatorFields=2&crossFieldBlendingBalance=1.5&layer0logPolar=false&layer0fieldResolutionHalvings=1&layer0kernelMaxDistance=47&layer0kernelSizeShrunkBy=edge&layer0kernelShrinkFactor=0.3&layer0ringcoupling=2.5_3_4.5_-3.5&layer0ringdistance=0.1_0.3_0.4_0.9&layer0ringwidth=0.05_0.08_0.12_0.15&layer0influenceFromOtherLayer=3&layer0oscillatorFrequency=0.6_2.6&layer0frequencyCorrelationPatternSize=100&layer1logPolar=true&layer1fieldResolutionHalvings=1&layer1kernelMaxDistance=12&layer1kernelSizeShrunkBy=nothing&layer1kernelShrinkFactor=0.3&layer1ringcoupling=5_-5_0_1.5&layer1ringdistance=0.1_0.3_0.6_0.9&layer1ringwidth=0.05_0.08_0.12_0.15&layer1influenceFromOtherLayer=4&layer1oscillatorFrequency=0.4_2.9&layer1frequencyCorrelationPatternSize=100"
			},
			{
				internalNameInStudy: "12_pregabalin_bubbles",
				url: "driftMaxDisplacement=0&driftPatternPlaybackSpeed=0.2&driftPatternSize=1280&deepDreamLayerIndex=0&useCustomVideo=false&customVideoPlaybackSpeed=0.5&deepDreamOpacity=0&deepLuminosityBlendOpacity=0.5&blendPatternPlaybackSpeed=0.2&blendPatternSize=1280&oscillatorOpacity=0.6&oscillatorColors=rainbow2x&numOscillatorFields=2&crossFieldBlendingBalance=1.5&layer0logPolar=false&layer0fieldResolutionHalvings=1&layer0kernelMaxDistance=47&layer0kernelSizeShrunkBy=edge&layer0kernelShrinkFactor=0.3&layer0ringcoupling=3.5_2.5_-3_0&layer0ringdistance=0.1_0.3_0.4_0.9&layer0ringwidth=0.05_0.08_0.12_0.15&layer0influenceFromOtherLayer=2.5&layer0oscillatorFrequency=0.6_2.6&layer0frequencyCorrelationPatternSize=100&layer1logPolar=true&layer1fieldResolutionHalvings=1&layer1kernelMaxDistance=46&layer1kernelSizeShrunkBy=nothing&layer1kernelShrinkFactor=0.3&layer1ringcoupling=0.5_3_5_-5&layer1ringdistance=0.1_0.3_0.5_0.9&layer1ringwidth=0.05_0.08_0.12_0.15&layer1influenceFromOtherLayer=-3&layer1oscillatorFrequency=0.4_2.9&layer1frequencyCorrelationPatternSize=100"
			},
			{
				internalNameInStudy: "04_dmt_5or6_at_once_edge_chaotic",
				url: "driftMaxDisplacement=0&driftPatternPlaybackSpeed=0.2&driftPatternSize=1280&deepDreamLayerIndex=0&useCustomVideo=false&customVideoPlaybackSpeed=0.5&deepDreamOpacity=0&deepLuminosityBlendOpacity=0.5&blendPatternPlaybackSpeed=0.2&blendPatternSize=1280&oscillatorOpacity=0.6&oscillatorColors=rainbow&numOscillatorFields=2&crossFieldBlendingBalance=1.5&layer0logPolar=false&layer0fieldResolutionHalvings=1&layer0kernelMaxDistance=47&layer0kernelSizeShrunkBy=edge&layer0kernelShrinkFactor=0.3&layer0ringcoupling=2.5_3_4.5_-3.5&layer0ringdistance=0.1_0.3_0.4_0.9&layer0ringwidth=0.05_0.08_0.12_0.15&layer0influenceFromOtherLayer=-0.5&layer0oscillatorFrequency=0.6_2.6&layer0frequencyCorrelationPatternSize=100&layer1logPolar=true&layer1fieldResolutionHalvings=3&layer1kernelMaxDistance=5&layer1kernelSizeShrunkBy=nothing&layer1kernelShrinkFactor=0.3&layer1ringcoupling=4_5_-3_-3&layer1ringdistance=0.1_0.3_0.6_0.9&layer1ringwidth=0.05_0.08_0.12_0.15&layer1influenceFromOtherLayer=3&layer1oscillatorFrequency=0.4_2.9&layer1frequencyCorrelationPatternSize=100"
			},
			{
				internalNameInStudy: "pool_bw_lp1_kr5_rc44-1",
				url: "http://replicationapp.hallucination-research.localhost/?photoIndex=4&driftMaxDisplacement=0&driftPatternPlaybackSpeed=0.2&driftPatternSize=1280&deepDreamLayerIndex=0&useCustomVideo=false&customVideoPlaybackSpeed=0.5&deepDreamOpacity=0&deepLuminosityBlendOpacity=0.5&blendPatternPlaybackSpeed=0.2&blendPatternSize=1280&oscillatorOpacity=0.25&oscillatorColors=black-white-soft-normal&numOscillatorFields=1&crossFieldBlendingBalance=1.5&layer0logPolar=true&layer0fieldResolutionHalvings=0&layer0kernelMaxDistance=5&layer0kernelSizeShrunkBy=nothing&layer0kernelShrinkFactor=0.3&layer0ringcoupling=3.5_3.5_-0.5_0&layer0ringdistance=0.1_0.3_0.6_0.9&layer0ringwidth=0.05_0.08_0.12_0.15&layer0influenceFromOtherLayer=0&layer0oscillatorFrequency=0.5_0.5&layer0frequencyCorrelationPatternSize=100&layer1logPolar=false&layer1fieldResolutionHalvings=1&layer1kernelMaxDistance=5&layer1kernelSizeShrunkBy=nothing&layer1kernelShrinkFactor=0.3&layer1ringcoupling=0_0_0_0&layer1ringdistance=0.1_0.3_0.6_0.9&layer1ringwidth=0.05_0.08_0.12_0.15&layer1influenceFromOtherLayer=0&layer1oscillatorFrequency=3_7&layer1frequencyCorrelationPatternSize=100"
			},
			{
				internalNameInStudy: "08_classic_highdose_lsd",
				url: "driftMaxDisplacement=0&driftPatternPlaybackSpeed=0.2&driftPatternSize=1280&deepDreamLayerIndex=0&useCustomVideo=false&customVideoPlaybackSpeed=0.5&deepDreamOpacity=0&deepLuminosityBlendOpacity=0.5&blendPatternPlaybackSpeed=0.2&blendPatternSize=1280&oscillatorOpacity=0.6&oscillatorColors=rainbow2x&numOscillatorFields=2&crossFieldBlendingBalance=1.5&layer0logPolar=true&layer0fieldResolutionHalvings=1&layer0kernelMaxDistance=47&layer0kernelSizeShrunkBy=edge&layer0kernelShrinkFactor=0.3&layer0ringcoupling=2.5_4_-3_0.5&layer0ringdistance=0.1_0.3_0.4_0.9&layer0ringwidth=0.05_0.08_0.12_0.15&layer0influenceFromOtherLayer=-2&layer0oscillatorFrequency=0.6_2.6&layer0frequencyCorrelationPatternSize=100&layer1logPolar=false&layer1fieldResolutionHalvings=1&layer1kernelMaxDistance=46&layer1kernelSizeShrunkBy=nothing&layer1kernelShrinkFactor=0.3&layer1ringcoupling=-4_3.5_3_-1&layer1ringdistance=0.1_0.3_0.6_0.9&layer1ringwidth=0.05_0.08_0.12_0.15&layer1influenceFromOtherLayer=0.5&layer1oscillatorFrequency=0.4_2.9&layer1frequencyCorrelationPatternSize=100"
			},
			{
				internalNameInStudy: "none",
				url: "driftMaxDisplacement=0.003&driftPatternPlaybackSpeed=0.2&driftPatternSize=470&deepDreamLayerIndex=1&useCustomVideo=false&customVideoPlaybackSpeed=0.5&deepDreamOpacity=1&deepLuminosityBlendOpacity=1&blendPatternPlaybackSpeed=1.54&blendPatternSize=1280&oscillatorOpacity=0.4&oscillatorColors=green-magenta-sharp&numOscillatorFields=2&crossFieldBlendingBalance=1.38&layer0logPolar=false&layer0fieldResolutionHalvings=0&layer0kernelMaxDistance=10&layer0kernelSizeShrunkBy=edge&layer0kernelShrinkFactor=0.3&layer0ringcoupling=5_5_-5_0&layer0ringdistance=0.1_0.3_0.6_0.9&layer0ringwidth=0.05_0.08_0.12_0.15&layer0influenceFromOtherLayer=-4&layer0oscillatorFrequency=0.5_1.4&layer0frequenciesVaryBy=edge&layer0frequencyCorrelationPatternSize=100&layer1logPolar=true&layer1fieldResolutionHalvings=1&layer1kernelMaxDistance=14&layer1kernelSizeShrunkBy=nothing&layer1kernelShrinkFactor=0.3&layer1ringcoupling=5_5_-5_0&layer1ringdistance=0.1_0.3_0.6_0.9&layer1ringwidth=0.05_0.08_0.12_0.15&layer1influenceFromOtherLayer=0&layer1oscillatorFrequency=0.1_1.6&layer1frequencyCorrelationPatternSize=100"
			},
		];

		const selectbox = document.getElementById("paramPreset");
		for (let i = 0; i < paramPresets.length; i++) {
			const option = document.createElement("option");
			option.value = i;
			option.appendChild(document.createTextNode(i+1));
			selectbox.appendChild(option);
		}
		document.getElementById('btnLoadParamPreset').addEventListener('click', (e) => {
			// const presetIndex = selectbox.selectedIndex;
			const presetIndex = selectbox.value;
			const queryString = paramPresets[presetIndex].url;
			// console.log(selectbox, presetIndex, queryString);
			importParametersFromUrlQueryString(queryString);
		});
	}




	// we've done a bunch of other stuff while the textures etc are downloading
	// but now it's time
	g_mainShader = await mainShaderPromise;
	g_initOscillatorsShader = await initOscillatorsShaderPromise;
	g_oscillationShader = await oscillationShaderPromise;
	g_drawKernelShader = await drawKernelShaderPromise;
	g_scalingShader = await scalingShaderPromise;
	// https://jameshfisher.com/2020/10/22/why-is-my-webgl-texture-upside-down/
	g_webglContext.pixelStorei(g_webglContext.UNPACK_FLIP_Y_WEBGL, true);
	g_displacementVideo = await g_displacementVideoPromise;
	g_displacementVideoTexture = createImageTexture(g_webglContext, g_displacementVideo, g_webglContext.MIRRORED_REPEAT);
	g_blendingPatternVideo = await g_blendingPatternVideoPromise;
	g_blendingPatternVideoTexture = createImageTexture(g_webglContext, g_blendingPatternVideo, g_webglContext.REPEAT);
	g_deepDreamVideo = await g_deepDreamVideoPromise;
	g_deepDreamVideoTexture = createImageTexture(g_webglContext, g_deepDreamVideo);
	g_tinyDummyTexture = createImageTexture(g_webglContext, await tinyDummyImagePromise)
	// console.log(g_deepDreamVideo, g_deepDreamVideoTexture);

	async function awaitMainImageDownloadsAndMakeTextures(mainImageDownloader, isFirstTime) {
		const mainImage = await mainImageDownloader.mainImagePromise;
		const mainImageDepth = await mainImageDownloader.mainImageDepthPromise;
		const mainImageEdges = await mainImageDownloader.mainImageEdgesPromise;
		if (!mainImage)
			console.error('main image missing', mainImageDownloader.sourcePhotoFilenames.base);
		else
			resizeCanvasToMatchMedia(mainImage);
		g_mainImageTexture = createImageTexture(g_webglContext, mainImage);
		g_mainImageDepthTexture = createImageTexture(g_webglContext, mainImageDepth);
		g_mainImageEdgeTexture = createImageTexture(g_webglContext, mainImageEdges);
		g_deepDreamTextures = [];
		for (let i = 0; i < mainImageDownloader.deepDreamPromises.length; i++) {
			g_deepDreamTextures.push(createImageTexture(g_webglContext, await mainImageDownloader.deepDreamPromises[i]));
		}
		console.log(mainImage, mainImageDepth, mainImageEdges, g_mainImageTexture, g_mainImageDepthTexture, g_mainImageEdgeTexture);
		
		// Generate auto-domains from image if enabled (async to avoid blocking)
		if (g_useAutoDomains && mainImage) {
			// Use requestIdleCallback or setTimeout to avoid blocking the main thread
			const generateDomainsAsync = () => {
				try {
					// Extract ImageData from loaded images
					const mainCanvas = document.createElement('canvas');
					mainCanvas.width = mainImage.width;
					mainCanvas.height = mainImage.height;
					const mainCtx = mainCanvas.getContext('2d');
					mainCtx.drawImage(mainImage, 0, 0);
					const mainImageData = mainCtx.getImageData(0, 0, mainCanvas.width, mainCanvas.height);
					
					let depthImageData = null;
					if (mainImageDepth) {
						const depthCanvas = document.createElement('canvas');
						depthCanvas.width = mainImageDepth.width;
						depthCanvas.height = mainImageDepth.height;
						const depthCtx = depthCanvas.getContext('2d');
						depthCtx.drawImage(mainImageDepth, 0, 0);
						depthImageData = depthCtx.getImageData(0, 0, depthCanvas.width, depthCanvas.height);
					}
					
					let edgeImageData = null;
					if (mainImageEdges) {
						const edgeCanvas = document.createElement('canvas');
						edgeCanvas.width = mainImageEdges.width;
						edgeCanvas.height = mainImageEdges.height;
						const edgeCtx = edgeCanvas.getContext('2d');
						edgeCtx.drawImage(mainImageEdges, 0, 0);
						edgeImageData = edgeCtx.getImageData(0, 0, edgeCanvas.width, edgeCanvas.height);
					}
					
					// Generate auto-domains
					g_autoDomainData = generateAutoDomains(mainImageData, depthImageData, edgeImageData);
					
					// Update domain mask texture to use auto-domains
					updateDomainMaskTexture();
					
					console.log(`Auto-domains generated: ${g_autoDomainData.numDomains} domains`);
				} catch (error) {
					console.error('Error generating auto-domains:', error);
				}
			};
			
			// Defer to next frame to avoid blocking initial load
			if (typeof requestIdleCallback !== 'undefined') {
				requestIdleCallback(generateDomainsAsync, { timeout: 1000 });
			} else {
				setTimeout(generateDomainsAsync, 100);
			}
		}

		{
			const infoParts = [];
			const sourceMeta = mainImageDownloader.sourcePhotoFilenames;
			if (sourceMeta.isUserUpload)
				infoParts.push('Using your upload');
			else if (sourceMeta.base)
				infoParts.push('Photo ' + sourceMeta.base);
			if (mainImage)
				infoParts.push(mainImage.width + '×' + mainImage.height);
			if (sourceMeta.depth) {
				if (mainImageDepth)
					infoParts.push('depth ok');
				else
					infoParts.push('missing depth (' + sourceMeta.depth + ')');
			}
			else if (mainImageDepth)
				infoParts.push('derived depth map');
			if (sourceMeta.edges) {
				if (mainImageEdges)
					infoParts.push('edge map ok');
				else
					infoParts.push('missing edges (' + sourceMeta.edges + ')');
			}
			else if (mainImageEdges)
				infoParts.push('derived edges');
			const deepDreamCount = sourceMeta.numDeepDreamLayers || g_deepDreamTextures.length;
			if (deepDreamCount)
				infoParts.push(deepDreamCount + ' DeepDream layers');
			document.getElementById("photoInfo").innerHTML = infoParts.join(' • ');
		}

		if (!isFirstTime) { // because the first time we _create_ the oscillator textures another way, we don't just _change_ them
			for (let iField in g_oscillatorFields)
				changeFrequenciesAndKernelShrinkageButKeepPhases(g_webglContext, iField);
			requestAnimationFrameNoDuplicate();
		}
	}
	await awaitMainImageDownloadsAndMakeTextures(mainImageDownloader, true);


function getActiveUniformNameSet(shader) {
	if (!shader._activeUniformNameSet) {
		const activeUniformCount = g_webglContext.getProgramParameter(shader, g_webglContext.ACTIVE_UNIFORMS);
		const names = new Set();
		for (let i = 0; i < activeUniformCount; i++) {
			const uniformInfo = g_webglContext.getActiveUniform(shader, i);
			if (!uniformInfo || !uniformInfo.name)
				continue;
			let name = uniformInfo.name;
			names.add(name);
			if (name.endsWith("[0]")) {
				const bracketIndex = name.indexOf('[');
				if (bracketIndex >= 0)
					names.add(name.slice(0, bracketIndex));
			}
		}
		shader._activeUniformNameSet = names;
	}
	return shader._activeUniformNameSet;
}

function isUniformActive(shader, uniformName) {
	const activeNames = getActiveUniformNameSet(shader);
	if (activeNames.has(uniformName))
		return true;
	const match = uniformName.match(/^(.*)\[\d+\]$/);
	if (match) {
		const baseName = match[1];
		if (activeNames.has(baseName) || activeNames.has(`${baseName}[0]`))
			return true;
	}
	return false;
}

// get uniform locations in shader and cache them, for performance
function cacheUniformLocation(shader, uniformName) {
		if (shader.uniformLocations === undefined)
			shader.uniformLocations = {};
	if (!isUniformActive(shader, uniformName)) {
		shader.uniformLocations[uniformName] = null;
		return;
	}
		// https://developer.mozilla.org/en-US/docs/Web/API/WebGLRenderingContext/getUniformLocation
		shader.uniformLocations[uniformName] = g_webglContext.getUniformLocation(shader, uniformName);
		if (shader.uniformLocations[uniformName] === null)
			console.warn(`getUniformLocation(${uniformName}) returned null. Maybe you just don't use it in your shader?`);
			// not neccessarily an error. if you have a uniform in your shader that you don't USE in your shader,
			// then getUniformLocation will return null for it. i guess the shader compilator has noticed unused uniforms
			// and removed them from the program.
	}
	cacheUniformLocation(g_mainShader, "uMainTex");
	cacheUniformLocation(g_mainShader, "uDeepDreamTex");
	cacheUniformLocation(g_mainShader, "uDisplacementMap");
	cacheUniformLocation(g_mainShader, "uBlendingPattern");
	cacheUniformLocation(g_mainShader, "uOscillatorTexRead");
	cacheUniformLocation(g_mainShader, "uNumOscillatorTextures");
	cacheUniformLocation(g_mainShader, "driftMaxDisplacementHoriz");
	cacheUniformLocation(g_mainShader, "driftMaxDisplacementVert");
	cacheUniformLocation(g_mainShader, "driftPatternScale");
	cacheUniformLocation(g_mainShader, "blendPatternScale");
	cacheUniformLocation(g_mainShader, "deepDreamOpacity");
	cacheUniformLocation(g_mainShader, "deepLuminosityBlendOpacity");
	cacheUniformLocation(g_mainShader, "oscillatorOpacity");
	cacheUniformLocation(g_mainShader, "oscillatorColors");
	cacheUniformLocation(g_mainShader, "crossFieldBlendingBalance");
	cacheUniformLocation(g_mainShader, "debugDrawKernelSize");
	cacheUniformLocation(g_mainShader, "debugDrawKernelSizeForField");
	cacheUniformLocation(g_mainShader, "kernelMaxDistance");
	cacheUniformLocation(g_mainShader, "debugDrawFrequency");
	cacheUniformLocation(g_mainShader, "frameIndex");
	cacheUniformLocation(g_mainShader, "frequencyRange");
	cacheUniformLocation(g_mainShader, "deltaTime");
	cacheUniformLocation(g_mainShader, "logPolarTransform");
	cacheUniformLocation(g_mainShader, "uDomainMaskTex");
	cacheUniformLocation(g_mainShader, "useDomains");
	cacheUniformLocation(g_mainShader, "numDomains");
	cacheUniformLocation(g_mainShader, "domainOrderParameter");
	cacheUniformLocation(g_mainShader, "domainChaosParameter");
	cacheUniformLocation(g_mainShader, "corticalDepressionIntensity");
	cacheUniformLocation(g_initOscillatorsShader, "uOscillatorTexRead");
	cacheUniformLocation(g_initOscillatorsShader, "uMainTex");
	cacheUniformLocation(g_initOscillatorsShader, "uDepthTex");
	cacheUniformLocation(g_initOscillatorsShader, "uEdgeTex");
	cacheUniformLocation(g_initOscillatorsShader, "frequencyRange");
	cacheUniformLocation(g_initOscillatorsShader, "frequenciesVaryBy");
	cacheUniformLocation(g_initOscillatorsShader, "frequencyCorrelationPatternSize");
	cacheUniformLocation(g_initOscillatorsShader, "kernelSizeShrunkBy");
	cacheUniformLocation(g_initOscillatorsShader, "kernelMaxShrinkDivisor");
	cacheUniformLocation(g_initOscillatorsShader, "logPolarTransform");
	// cacheUniformLocation(g_oscillationShader, "couplingSmallWorld");
	cacheUniformLocation(g_oscillationShader, "uOscillatorTexRead");
	cacheUniformLocation(g_oscillationShader, "uOscillatorTexOther");
	// cacheUniformLocation(g_oscillationShader, "uOscillatorTexSize");
	cacheUniformLocation(g_oscillationShader, "kernelMaxDistance");
	cacheUniformLocation(g_oscillationShader, "kernelRingCoupling");
	cacheUniformLocation(g_oscillationShader, "kernelRingDistances");
	cacheUniformLocation(g_oscillationShader, "kernelRingWidths");
	cacheUniformLocation(g_oscillationShader, "couplingToOtherField");
	// cacheUniformLocation(g_oscillationShader, "debugDrawFilterKernel");
	cacheUniformLocation(g_oscillationShader, "deltaTime");
	cacheUniformLocation(g_oscillationShader, "logPolarTransformMe");
	cacheUniformLocation(g_oscillationShader, "logPolarTransformOther");
	cacheUniformLocation(g_oscillationShader, "corticalDepressionIntensity");
	// Domain-related uniforms
	cacheUniformLocation(g_oscillationShader, "uDomainMaskTex");
	cacheUniformLocation(g_oscillationShader, "useDomains");
	cacheUniformLocation(g_oscillationShader, "numDomains");
	for (let i = 0; i < 8; i++) {
		cacheUniformLocation(g_oscillationShader, `domainKernelRingCoupling[${i}]`);
		cacheUniformLocation(g_oscillationShader, `domainKernelRingDistances[${i}]`);
		cacheUniformLocation(g_oscillationShader, `domainKernelRingWidths[${i}]`);
		cacheUniformLocation(g_oscillationShader, `domainFrequencyMin[${i}]`);
		cacheUniformLocation(g_oscillationShader, `domainFrequencyMax[${i}]`);
	}
	cacheUniformLocation(g_oscillationShader, "domainOrderParameter");
	cacheUniformLocation(g_oscillationShader, "domainChaosParameter");
	cacheUniformLocation(g_drawKernelShader, "uOscillatorTexSize");
	cacheUniformLocation(g_drawKernelShader, "kernelMaxDistance");
	cacheUniformLocation(g_drawKernelShader, "kernelRingCoupling");
	cacheUniformLocation(g_drawKernelShader, "kernelRingDistances");
	cacheUniformLocation(g_drawKernelShader, "kernelRingWidths");
	cacheUniformLocation(g_scalingShader, "uTexture");
	// console.log(g_mainShader.uniformLocations);
	// console.log(g_oscillationShader.uniformLocations);
	// console.log(g_drawKernelShader.uniformLocations);
	// console.log(g_scalingShader.uniformLocations);



	// initializing oscillator textures is done with the help of the shader init-oscillators.frag,
	// therefore do this _after_ we have cached the uniforms, not before
	{
		g_oscillatorFramebuffer = g_webglContext.createFramebuffer();
		g_webglContext.pixelStorei(g_webglContext.UNPACK_FLIP_Y_WEBGL, false);
		const oscillatorSourcePixels = createOscillatorSourcePixels(canvas.width, canvas.height);
		for (let iField in g_oscillatorFields) {
			g_oscillatorFields[iField].oscillatorTextures = [];
			g_oscillatorFields[iField].oscillatorTextures[0] = createOscillatorTexture(g_webglContext, canvas.width, canvas.height, oscillatorSourcePixels);
			g_oscillatorFields[iField].oscillatorTextures[1] = createOscillatorTexture(g_webglContext, canvas.width, canvas.height, oscillatorSourcePixels);
			// console.log(g_oscillatorFields[iField].oscillatorTextures[0], g_oscillatorFields[iField].oscillatorTextures[1]);
			g_oscillatorFields[iField].oscillatorTextureWidth = canvas.width;
		}
		g_webglContext.pixelStorei(g_webglContext.UNPACK_FLIP_Y_WEBGL, true);
		for (let iField in g_oscillatorFields)
			changeFrequenciesAndKernelShrinkageButKeepPhases(g_webglContext, iField);
	}


	if (window.location.search) {
		const queryString = decodeURI(window.location.search.substr(1));
		// console.log(queryString);
		// importParametersFromJsonString(queryString);
		importParametersFromUrlQueryString(queryString);
	}

	console.log('setup done');

	window.requestAnimationFrame(drawWebGl);
	// always do this _once_ even if g_isPlaying is false, so we draw a picture
	// but then if g_isPlaying is false we won't keep requesting more frames
}

function clamp(val, min, max) {
	console.assert(!Number.isNaN(val), "!Number.isNaN(val)");
	return Math.max(Math.min(val, max), min);
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

function downloadVideo(url) {
	const video = document.createElement("video");

	// document.body.appendChild(video);

	video.playsInline = true;
	video.muted = true;
	video.loop = true;

	const prom = new Promise((resolve, reject) => {
		// Waiting for these 2 events ensures there is data in the video
		let playing = false;
		let timeupdate = false;

		function playingListener() {
			// console.log("playingListener", playing, timeupdate);
			playing = true;
			checkReady();
		}
		function timeupdateListener() {
			// console.log("timeupdateListener", playing, timeupdate);
			timeupdate = true;
			checkReady();
		}
		video.addEventListener("playing", playingListener);
		video.addEventListener("timeupdate", timeupdateListener);

		video.onerror = function() {
			console.error("video error", url);
			reject();
		};

		video.src = url;
		video.play();

		function checkReady() {
			// console.log("checkReady", playing, timeupdate);
			if (playing && timeupdate) {
				video.removeEventListener("playing", playingListener);
				video.removeEventListener("timeupdate", timeupdateListener);
				resolve(video);
			}
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

// call after bindTexture
function updateVideoTexture(gl, video) {
	gl.texImage2D(gl.TEXTURE_2D,
		0, // level
		gl.RGBA, // internalFormat
		gl.RGBA, // srcFormat
		gl.UNSIGNED_BYTE, // srcType
		video);
}


function createOscillatorSourcePixels(width, height) {
	// should we use angles or complex numbers for the phase? complex numbers!
	// why? because otherwise gl.LINEAR interpolation will look bad, when the graphics card takes the average of say 0.1 and 1.9*PI, the average will be PI instead of 0

	// #TODO: decide if we should use bytes, floats, or half-floats for the oscillators
	// decide based on both convenience and performance

	const oscillatorSourcePixels = new Float32Array(width * height * 4); // 4 channels: RGBA
	for (let y = 0; y < height; y++) {
		for (let x = 0; x < width; x++) {
			const index = (x + (y * width)) * 4;
			const phase = Math.random() * 2 * Math.PI; // random noise
			// const phase = 0;
			// const phase = (x / 20) % (2 * Math.PI); // plane waves
			// const r = Math.min(1, 2.5 * edgeImageData[index] / 255);
			// const phase = r % (2 * Math.PI); // waves emitted from image edges
			// const phase = (Math.random() + r)  * Math.PI; // noise plus waves emitted from image edges
			// option 1: store phase as angle between 0 and 2pi
			// oscillatorSourcePixels[index + 0] = phase;
			// oscillatorSourcePixels[index + 1] = 0; // unused
			// option 2: store phase as complex number. leads to better gl.LINEAR interpolation as we scale the texture
			oscillatorSourcePixels[index + 0] = Math.cos(phase);
			oscillatorSourcePixels[index + 1] = Math.sin(phase);
			oscillatorSourcePixels[index + 2] = 1.0; // written by init-oscillators.frag
			oscillatorSourcePixels[index + 3] = 1.0; // written by init-oscillators.frag
		}
	}

	return oscillatorSourcePixels;
}

function createOscillatorTexture(gl, width, height, sourcePixels) {
	// console.log('createOscillatorTexture(', width, height, ')');
	// In WebGL 1, format must be the same as internalformat.
	// In WebGL 2, the allowed combinations are listed in this table:
	// https://registry.khronos.org/webgl/specs/latest/2.0/#TEXTURE_TYPES_FORMATS_FROM_DOM_ELEMENTS_TABLE
	// https://webgl2fundamentals.org/webgl/lessons/webgl-data-textures.html
	const texture = gl.createTexture();
	gl.bindTexture(gl.TEXTURE_2D, texture);
	gl.texImage2D(gl.TEXTURE_2D,
		0, // level
		gl.RGBA32F, // internalFormat
		width,
		height,
		0,
		gl.RGBA, // format
		gl.FLOAT, // srcType
		sourcePixels);

	// Turn off mipmaps and set wrapping to clamp to edge so it
	// will work regardless of the dimensions
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

	return texture;
}

function unifyOscillatorPhases(gl, iField) {
	resetOscillatorPhases(gl, iField, function(osciX, osciY) {
		return 0;
	});
}

function unifyAllPhases(gl) {
	for (let iField = 0; iField < g_params.numOscillatorFields; iField++) {
		resetOscillatorPhases(gl, iField, function(osciX, osciY) {
			return 0;
		});
	}
}

function randomizeOscillatorPhases(gl, iField) {
	resetOscillatorPhases(gl, iField, function(osciX, osciY) {
		return Math.random() * 2 * Math.PI;
	});
}

function randomizeAllPhases(gl) {
	for (let iField = 0; iField < g_params.numOscillatorFields; iField++) {
		randomizeOscillatorPhases(gl, iField);
	}
	requestAnimationFrameNoDuplicate();
}

function resetOscillatorPhasesToPlaneWaves(gl, iField, desiredWaveTravelAngleRadians) {

	const oscfield = g_oscillatorFields[iField];
	// another complication is that the oscillator texture might no longer have the same size as when we first created it
	const scaleFactorInActualCurrentUse = g_webglContext.canvas.width / oscfield.oscillatorTextureWidth;
	const osciWidth = Math.round(g_webglContext.canvas.width / scaleFactorInActualCurrentUse);
	const osciHeight = Math.round(g_webglContext.canvas.height / scaleFactorInActualCurrentUse);

	const desiredWaveLengthInPixels = osciWidth / 20;
	const vecMultiplier = adjustWavelengthAndAngleForPerfectWraparound(osciWidth, osciHeight, desiredWaveLengthInPixels, desiredWaveTravelAngleRadians);

	resetOscillatorPhases(gl, iField, function(osciX, osciY, osciWidth, osciHeight) {
		return (2 * Math.PI) * (
			osciX * vecMultiplier.x +
			osciY * vecMultiplier.y
		);
	});
}

// function resetOscillatorPhasesToEdgeWaves(gl, iField) {
// 	resetOscillatorPhases(gl, iField, function(osciX, osciY, osciWidth, osciHeight) {
// 		const r = Math.min(1, 2.5 * edgeImageData[index] / 255);
// 		const phase = r % (2 * Math.PI); // waves emitted from image edges
// 		return phase;
// 	});
// }

function resetOscillatorPhases(gl, iField, callbackFunc) {

	const oscfield = g_oscillatorFields[iField];

	// it's a bit complicated to change the oscillator phases on the fly
	// we need to read back the oscillator texture from the graphics card,
	// keep its b and a channels, that contain other things,
	// but change only its r and b channels, that contain the phase info
	// then write it back to the graphics card

	// another complication is that the oscillator texture might no longer have the same size as when we first created it
	const scaleFactorInActualCurrentUse = g_webglContext.canvas.width / oscfield.oscillatorTextureWidth;
	const osciWidth = Math.round(g_webglContext.canvas.width / scaleFactorInActualCurrentUse);
	const osciHeight = Math.round(g_webglContext.canvas.height / scaleFactorInActualCurrentUse);

	// how to read back the texture from the graphics card?
	// attach the texture to a framebuffer, then call readPixels on the framebuffer
	const oscillatorTexRead = oscfield.oscillatorTextures[oscfield.latestIndex];
	gl.bindFramebuffer(gl.FRAMEBUFFER, g_oscillatorFramebuffer);
	gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, oscillatorTexRead, 0);
	const canRead = (gl.checkFramebufferStatus(gl.FRAMEBUFFER) == gl.FRAMEBUFFER_COMPLETE);
	// For textures of format = gl.RGBA, type = gl.UNSIGNED_BYTE canRead should always be true. For other formats and types canRead might be false.
	// gl.RGBA and gl.FLOAT _should_ work in WebGL 2 according to https://stackoverflow.com/questions/17981163/webgl-read-pixels-from-floating-point-render-target
	let existingOscillatorPixels;
	if (canRead) {
		// https://registry.khronos.org/webgl/specs/latest/2.0/#readpixels
		// https://developer.mozilla.org/en-US/docs/Web/API/WebGLRenderingContext/readPixels
		existingOscillatorPixels = new Float32Array(osciWidth * osciHeight * 4); // 4 channels: RGBA
		gl.readPixels(0, 0, osciWidth, osciHeight, gl.RGBA, gl.FLOAT, existingOscillatorPixels);
	}
	gl.bindFramebuffer(gl.FRAMEBUFFER, null);
	if (canRead) {
		g_webglContext.pixelStorei(g_webglContext.UNPACK_FLIP_Y_WEBGL, false);

		for (let osciY = 0; osciY < osciHeight; osciY++) {
			for (let osciX = 0; osciX < osciWidth; osciX++) {
				// let canvasX = osciX * scaleFactorInActualCurrentUse;
				// let canvasY = osciY * scaleFactorInActualCurrentUse;
				const index = (osciX + (osciY * osciWidth)) * 4;
				const phase = callbackFunc(osciX, osciY, osciWidth, osciHeight);
				existingOscillatorPixels[index + 0] = Math.cos(phase);
				existingOscillatorPixels[index + 1] = Math.sin(phase);
				// existingOscillatorPixels[index + 2] unchanged
				// existingOscillatorPixels[index + 3] unchanged
			}
		}

		gl.bindTexture(gl.TEXTURE_2D, oscillatorTexRead);
		gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, osciWidth, osciHeight, 0, gl.RGBA, gl.FLOAT, existingOscillatorPixels);

		g_webglContext.pixelStorei(g_webglContext.UNPACK_FLIP_Y_WEBGL, true);
	}
	return canRead;
}

// ---------------------- Custom uploads and image processing helpers ----------------------

function resizeCanvasToMatchMedia(image) {
	if (!image)
		return;
	const canvas = document.getElementById('glCanvas');
	if (!canvas)
		return;
	let targetWidth = image.width || canvas.width || MIN_CANVAS_WIDTH;
	let targetHeight = image.height || canvas.height || (MIN_CANVAS_WIDTH * 9 / 16);
	if (targetWidth > MAX_CANVAS_WIDTH) {
		const scaleDown = MAX_CANVAS_WIDTH / targetWidth;
		targetWidth = Math.round(targetWidth * scaleDown);
		targetHeight = Math.round(targetHeight * scaleDown);
	}
	if (targetHeight > MAX_CANVAS_HEIGHT) {
		const scaleDown = MAX_CANVAS_HEIGHT / targetHeight;
		targetWidth = Math.max(4, Math.round(targetWidth * scaleDown));
		targetHeight = Math.round(targetHeight * scaleDown);
	}
	targetWidth = Math.max(64, targetWidth);
	targetHeight = Math.max(64, targetHeight);
	if (canvas.width === targetWidth && canvas.height === targetHeight)
		return;
	canvas.width = targetWidth;
	canvas.height = targetHeight;
	// Update domain mask texture when canvas resizes
	if (g_domains.length > 0) {
		updateDomainMaskTexture();
	}
	canvas.style.aspectRatio = targetWidth + ' / ' + targetHeight;
	canvas.style.height = 'auto';
	if (g_oscillatorFields[0] && g_oscillatorFields[0].oscillatorTextures.length)
		reinitializeOscillatorTexturesForCanvas(targetWidth, targetHeight);
	// Update domain mask after textures are reinitialized
	if (g_domains.length > 0 && g_webglContext) {
		updateDomainMaskTexture();
	}
}

function reinitializeOscillatorTexturesForCanvas(newWidth, newHeight) {
	if (!g_webglContext)
		return;
	const oscillatorSourcePixels = createOscillatorSourcePixels(newWidth, newHeight);
	g_webglContext.pixelStorei(g_webglContext.UNPACK_FLIP_Y_WEBGL, false);
	for (let iField in g_oscillatorFields) {
		const field = g_oscillatorFields[iField];
		if (!field.oscillatorTextures)
			field.oscillatorTextures = [];
		for (let i = 0; i < field.oscillatorTextures.length; i++) {
			if (field.oscillatorTextures[i])
				g_webglContext.deleteTexture(field.oscillatorTextures[i]);
		}
		field.oscillatorTextures[0] = createOscillatorTexture(g_webglContext, newWidth, newHeight, oscillatorSourcePixels);
		field.oscillatorTextures[1] = createOscillatorTexture(g_webglContext, newWidth, newHeight, oscillatorSourcePixels);
		field.latestIndex = 0;
		field.oscillatorTextureWidth = newWidth;
	}
	g_webglContext.pixelStorei(g_webglContext.UNPACK_FLIP_Y_WEBGL, true);
	for (let iField in g_oscillatorFields)
		changeFrequenciesAndKernelShrinkageButKeepPhases(g_webglContext, iField);
	requestAnimationFrameNoDuplicate();
}

function initCustomUploadControls() {
	const photoInput = document.getElementById('customPhotoInput');
	const depthInput = document.getElementById('customDepthInput');
	const edgeSelect = document.getElementById('customEdgeStrategy');
	const depthSelect = document.getElementById('customDepthStrategy');
	const thresholdInput = document.getElementById('customEdgeThreshold');
	const regenerateBtn = document.getElementById('btnRegenerateCustomMaps');
	const useBtn = document.getElementById('btnUseCustomUpload');
	const previewBase = document.getElementById('customPreviewBase');
	const previewEdge = document.getElementById('customPreviewEdge');
	const previewDepth = document.getElementById('customPreviewDepth');
	if (!photoInput || !edgeSelect || !depthSelect || !thresholdInput || !regenerateBtn || !useBtn)
		return;
	g_customPreviewTargets = {
		base: previewBase,
		edge: previewEdge,
		depth: previewDepth
	};
	renderCustomUploadPreviews();

	// Processing state (must be declared before functions that use it)
	let regenerateTimeout = null;
	let isProcessing = false;

	const enableButtons = function(enable) {
		if (!isProcessing) {
			regenerateBtn.disabled = !enable || !g_customUploadState.baseImageData;
			useBtn.disabled = !enable || !g_customUploadState.baseCanvas;
		}
	};
	
	const setProcessingState = function(processing) {
		isProcessing = processing;
		regenerateBtn.disabled = processing || !g_customUploadState.baseImageData;
		useBtn.disabled = processing || !g_customUploadState.baseCanvas;
		debugCustomUpload('setProcessingState', {
			processing,
			edgeThreshold: g_customUploadState.edgeThreshold,
			hasBaseImage: !!g_customUploadState.baseImageData
		});
		if (processing) {
			regenerateBtn.textContent = 'Processing…';
		} else {
			regenerateBtn.textContent = 'Rebuild maps';
		}
	};
	
	const refreshAlgorithms = function(showStatus = true) {
		debugCustomUpload('refreshAlgorithms:start', {
			showStatus,
			edgeStrategy: edgeSelect.value,
			depthStrategy: depthSelect.value,
			edgeThreshold: g_customUploadState.edgeThreshold,
			isProcessing
		});
		if (!g_customUploadState.baseImageData) {
			debugCustomUpload('refreshAlgorithms:skipped-no-base');
			return;
		}
		
		if (isProcessing) {
			// Already processing, queue another update
			if (regenerateTimeout)
				clearTimeout(regenerateTimeout);
			regenerateTimeout = setTimeout(() => refreshAlgorithms(showStatus), 100);
			return;
		}
		
		setProcessingState(true);
		g_customUploadState.edgeStrategy = edgeSelect.value;
		g_customUploadState.depthStrategy = depthSelect.value;
		g_customUploadState.edgeThreshold = parseFloat(thresholdInput.value);
		
		// Run processing async to avoid blocking UI
		setTimeout(() => {
			try {
				debugCustomUpload('refreshAlgorithms:process');
				regenerateDerivedMaps();
				if (showStatus) {
					setCustomUploadStatus('Maps updated.');
				}
				debugCustomUpload('refreshAlgorithms:success', {
					edgeCanvas: describeDrawable(g_customUploadState.edgeCanvas),
					depthCanvas: describeDrawable(g_customUploadState.depthCanvas)
				});
			} catch (err) {
				console.error('Error regenerating maps:', err);
				setCustomUploadStatus('Error: ' + (err.message || 'Failed to regenerate maps'));
				debugCustomUpload('refreshAlgorithms:error', err);
			} finally {
				setProcessingState(false);
				debugCustomUpload('refreshAlgorithms:complete');
			}
		}, 0);
	};

	photoInput.addEventListener('change', async (event) => {
		const file = event.target.files && event.target.files[0];
		if (!file)
			return;
		setProcessingState(true);
		setCustomUploadStatus('Loading base image…');
		try {
			const image = await loadImageFromFile(file);
			debugCustomUpload('photoInput:loaded', {
				fileName: file.name || 'upload',
				image: describeDrawable(image)
			});
			setCustomUploadBaseImage(image, file.name || 'upload');
			if (depthInput) {
				depthInput.value = '';
				g_customUploadState.depthOverrideCanvas = null;
			}
			// Process maps async
			setTimeout(() => {
				try {
					refreshAlgorithms(false);
					setCustomUploadStatus('Image ready. Adjust settings or click "Use upload" to apply.');
				} catch (err) {
					console.error('Error processing image:', err);
					setCustomUploadStatus('Error processing image: ' + (err.message || 'Unknown error'));
					setProcessingState(false);
				}
			}, 0);
		}
		catch (err) {
			console.error(err);
			setCustomUploadStatus('Failed to load image: ' + (err.message || 'Unknown error'));
			setProcessingState(false);
			enableButtons(false);
		}
	});

	depthInput.addEventListener('change', async (event) => {
		if (!g_customUploadState.baseImageData) {
			depthInput.value = '';
			setCustomUploadStatus('Select a base image before adding a depth map.');
			return;
		}
		const file = event.target.files && event.target.files[0];
		if (!file) {
			g_customUploadState.depthOverrideCanvas = null;
			setTimeout(() => {
				regenerateDerivedMaps();
				setCustomUploadStatus('Depth override cleared.');
			}, 0);
			return;
		}
		setCustomUploadStatus('Loading depth map…');
		try {
			const image = await loadImageFromFile(file);
			debugCustomUpload('depthInput:loaded', {
				fileName: file.name || 'depth-upload',
				image: describeDrawable(image)
			});
			g_customUploadState.depthOverrideCanvas = resampleImageToCanvas(image, g_customUploadState.baseImageData.width, g_customUploadState.baseImageData.height);
			setTimeout(() => {
				regenerateDerivedMaps();
				setCustomUploadStatus('Depth override applied.');
			}, 0);
		}
		catch (err) {
			console.error(err);
			setCustomUploadStatus('Failed to load depth map: ' + (err.message || 'Unknown error'));
		}
	});

	// Debounced threshold updates (wait for user to stop sliding)
	let thresholdDebounce = null;
	edgeSelect.addEventListener('change', () => refreshAlgorithms(true));
	depthSelect.addEventListener('change', () => refreshAlgorithms(true));
	thresholdInput.addEventListener('input', () => {
		const newThreshold = parseFloat(thresholdInput.value);
		g_customUploadState.edgeThreshold = newThreshold;
		debugCustomUpload('edgeThreshold:input', { newThreshold });
		if (g_customUploadState.baseImageData) {
			// Debounce: wait 300ms after user stops sliding
			if (thresholdDebounce)
				clearTimeout(thresholdDebounce);
			setCustomUploadStatus('Adjusting threshold…');
			thresholdDebounce = setTimeout(() => {
				debugCustomUpload('edgeThreshold:apply', { newThreshold });
				refreshAlgorithms(true);
			}, 300);
		}
	});
	regenerateBtn.addEventListener('click', () => refreshAlgorithms(true));
	useBtn.addEventListener('click', async () => {
		if (!g_applyCustomUploadHandler) {
			setCustomUploadStatus('Upload handler unavailable.');
			return;
		}
		if (!g_customUploadState.baseCanvas) {
			setCustomUploadStatus('Select an image before applying.');
			return;
		}
		if (isProcessing) {
			setCustomUploadStatus('Please wait for processing to complete.');
			return;
		}
		setProcessingState(true);
		setCustomUploadStatus('Applying upload…');
		try {
			await g_applyCustomUploadHandler();
			setCustomUploadStatus('Custom image active.');
		} catch (err) {
			console.error(err);
			setCustomUploadStatus('Failed to apply upload: ' + (err.message || 'Unknown error'));
		} finally {
			setProcessingState(false);
		}
	});

	enableButtons(false);
	setCustomUploadStatus('Select an image to enable uploads.');
}

function createInitialCustomUploadState() {
	return {
		fileName: '',
		baseCanvas: null,
		baseImageData: null,
		grayscale: null,
		sobelMagnitude: null,
		distanceField: null,
		edgeCanvas: null,
		edgeImageData: null,
		depthCanvas: null,
		depthOverrideCanvas: null,
		edgeStrategy: 'sobel',
		depthStrategy: 'luma',
		edgeThreshold: 0.35
	};
}

function setCustomUploadStatus(msg) {
	const statusEl = document.getElementById('customUploadStatus');
	if (statusEl)
		statusEl.textContent = msg;
}

function renderCustomUploadPreviews() {
	drawPreviewCanvas(g_customPreviewTargets.base, g_customUploadState.baseCanvas);
	drawPreviewCanvas(g_customPreviewTargets.edge, g_customUploadState.edgeCanvas);
	drawPreviewCanvas(g_customPreviewTargets.depth, g_customUploadState.depthCanvas || g_customUploadState.depthOverrideCanvas);
}

function drawPreviewCanvas(canvas, source) {
	if (!canvas) {
		debugCustomUpload('drawPreviewCanvas:skipped-no-canvas');
		return;
	}
	debugCustomUpload('drawPreviewCanvas:start', {
		target: describeDrawable(canvas),
		source: describeDrawable(source)
	});
	const ctx = canvas.getContext('2d');
	const { width, height } = canvas;
	ctx.save();
	ctx.fillStyle = '#101010';
	ctx.fillRect(0, 0, width, height);
	if (!source) {
		ctx.fillStyle = '#6c757d';
		ctx.font = '10px monospace';
		ctx.textAlign = 'center';
		ctx.textBaseline = 'middle';
		ctx.fillText('No data', width / 2, height / 2);
		ctx.restore();
		debugCustomUpload('drawPreviewCanvas:no-source', {
			target: describeDrawable(canvas)
		});
		return;
	}
	const srcWidth = source.width || source.videoWidth || 1;
	const srcHeight = source.height || source.videoHeight || 1;
	const scale = Math.min(width / srcWidth, height / srcHeight);
	const drawWidth = srcWidth * scale;
	const drawHeight = srcHeight * scale;
	const offsetX = (width - drawWidth) / 2;
	const offsetY = (height - drawHeight) / 2;
	ctx.imageSmoothingEnabled = true;
	ctx.imageSmoothingQuality = 'high';
	ctx.drawImage(source, offsetX, offsetY, drawWidth, drawHeight);
	ctx.restore();
	debugCustomUpload('drawPreviewCanvas:complete', {
		target: describeDrawable(canvas),
		source: describeDrawable(source),
		drawWidth,
		drawHeight
	});
}

function loadImageFromFile(file) {
	return new Promise((resolve, reject) => {
		const reader = new FileReader();
		reader.onload = function(evt) {
			const img = new Image();
			img.onload = function() {
				resolve(img);
			};
			img.onerror = reject;
			img.src = evt.target.result;
		};
		reader.onerror = reject;
		reader.readAsDataURL(file);
	});
}

function resampleImageToCanvas(image, width, height) {
	const canvas = document.createElement('canvas');
	canvas.width = width;
	canvas.height = height;
	const ctx = canvas.getContext('2d');
	ctx.drawImage(image, 0, 0, width, height);
	return canvas;
}

function setCustomUploadBaseImage(image, fileName) {
	debugCustomUpload('setCustomUploadBaseImage:start', {
		fileName,
		image: describeDrawable(image)
	});
	const preservedEdgeStrategy = g_customUploadState.edgeStrategy;
	const preservedDepthStrategy = g_customUploadState.depthStrategy;
	const preservedThreshold = g_customUploadState.edgeThreshold;
	g_customUploadState = createInitialCustomUploadState();
	g_customUploadState.edgeStrategy = preservedEdgeStrategy;
	g_customUploadState.depthStrategy = preservedDepthStrategy;
	g_customUploadState.edgeThreshold = preservedThreshold;

	const baseCanvas = resampleImageToCanvas(image, image.width, image.height);
	const ctx = baseCanvas.getContext('2d');
	const baseImageData = ctx.getImageData(0, 0, baseCanvas.width, baseCanvas.height);
	const grayscale = computeGrayscale(baseImageData);
	const sobelMagnitude = computeSobelMagnitude(grayscale, baseCanvas.width, baseCanvas.height);
	const normalizedSobel = normalizeFloatArray(sobelMagnitude);
	g_customUploadState.baseCanvas = baseCanvas;
	g_customUploadState.fileName = fileName;
	g_customUploadState.baseImageData = baseImageData;
	g_customUploadState.grayscale = grayscale;
	g_customUploadState.sobelMagnitude = normalizedSobel;
	g_customUploadState.depthOverrideCanvas = null;
	renderCustomUploadPreviews();
	debugCustomUpload('setCustomUploadBaseImage:done', {
		baseCanvas: describeDrawable(baseCanvas),
		edgeStrategy: g_customUploadState.edgeStrategy,
		depthStrategy: g_customUploadState.depthStrategy,
		edgeThreshold: g_customUploadState.edgeThreshold
	});
}

function regenerateDerivedMaps() {
	if (!g_customUploadState.baseImageData)
		return;
	const width = g_customUploadState.baseImageData.width;
	const height = g_customUploadState.baseImageData.height;
	const threshold = g_customUploadState.edgeThreshold || 0.35;
	debugCustomUpload('regenerateDerivedMaps:start', {
		width,
		height,
		threshold,
		edgeStrategy: g_customUploadState.edgeStrategy,
		depthStrategy: g_customUploadState.depthStrategy
	});
	let edgeStrength = g_customUploadState.sobelMagnitude;
	let coherence = null;
	if (g_customUploadState.edgeStrategy === 'kuramoto') {
		coherence = generateKuramotoCoherence(g_customUploadState.grayscale, width, height);
		edgeStrength = new Float32Array(coherence.length);
		for (let i = 0; i < coherence.length; i++)
			edgeStrength[i] = 1 - coherence[i];
	}
	const seedMask = buildSeedMask(edgeStrength, threshold);
	const distanceField = jumpFloodDistanceTransform(seedMask, width, height);
	let edgeImageData;
	if (g_customUploadState.edgeStrategy === 'jumpFlood')
		edgeImageData = distanceFieldToImageData(distanceField, width, height);
	else if (g_customUploadState.edgeStrategy === 'kuramoto' && coherence)
		edgeImageData = coherenceToImageData(coherence, width, height);
	else
		edgeImageData = edgeStrengthToImageData(edgeStrength, width, height);
	g_customUploadState.distanceField = distanceField;
	g_customUploadState.edgeCanvas = imageDataToCanvas(edgeImageData);
	g_customUploadState.edgeImageData = edgeImageData;

	if (g_customUploadState.depthOverrideCanvas)
		g_customUploadState.depthCanvas = g_customUploadState.depthOverrideCanvas;
	else {
		const depthImageData = buildDepthImageData(distanceField);
		g_customUploadState.depthCanvas = imageDataToCanvas(depthImageData);
	}
	renderCustomUploadPreviews();
	debugCustomUpload('regenerateDerivedMaps:complete', {
		edgeCanvas: describeDrawable(g_customUploadState.edgeCanvas),
		depthCanvas: describeDrawable(g_customUploadState.depthCanvas),
		distanceFieldPresent: !!g_customUploadState.distanceField
	});
}

function buildCustomSourceDescriptor() {
	if (!g_customUploadState.baseCanvas) {
		setCustomUploadStatus('Select an image before applying.');
		return null;
	}
	if (!g_customUploadState.edgeCanvas)
		regenerateDerivedMaps();
	const deepDreamLayers = generatePlaceholderDeepDreamLayers(g_customUploadState.baseCanvas, 4);
	return {
		base: g_customUploadState.fileName || 'upload',
		baseImageElement: g_customUploadState.baseCanvas,
		depthImageElement: g_customUploadState.depthCanvas || g_customUploadState.baseCanvas,
		edgeImageElement: g_customUploadState.edgeCanvas || g_customUploadState.baseCanvas,
		numDeepDreamLayers: Math.max(1, deepDreamLayers.length),
		deepDreamLayerElements: deepDreamLayers,
		isUserUpload: true
	};
}

function buildDepthImageData(distanceField) {
	const width = g_customUploadState.baseImageData.width;
	const height = g_customUploadState.baseImageData.height;
	let sourceValues;
	if (g_customUploadState.depthStrategy === 'edgeDistance' && distanceField)
		sourceValues = distanceField;
	else if (g_customUploadState.depthStrategy === 'flat') {
		sourceValues = new Float32Array(width * height);
		for (let i = 0; i < sourceValues.length; i++)
			sourceValues[i] = 0.5;
	}
	else
		sourceValues = g_customUploadState.grayscale;
	return floatArrayToImageData(sourceValues, width, height, false, 1.0);
}

function generatePlaceholderDeepDreamLayers(baseCanvas, count) {
	const layers = [];
	if (!baseCanvas)
		return layers;
	const layerCount = Math.max(1, count || 1);
	for (let i = 0; i < layerCount; i++) {
		const scale = Math.pow(0.65, i + 1);
		const width = Math.max(32, Math.round(baseCanvas.width * scale));
		const height = Math.max(32, Math.round(baseCanvas.height * scale));
		const canvas = document.createElement('canvas');
		canvas.width = width;
		canvas.height = height;
		const ctx = canvas.getContext('2d');
		ctx.filter = 'blur(' + (1 + i) + 'px) saturate(' + (1 - i * 0.12) + ')';
		ctx.drawImage(baseCanvas, 0, 0, width, height);
		layers.push(canvas);
	}
	if (!layers.length)
		layers.push(baseCanvas);
	return layers;
}

function buildSeedMask(edgeStrength, threshold) {
	const mask = new Uint8Array(edgeStrength.length);
	let hasSeed = false;
	for (let i = 0; i < edgeStrength.length; i++) {
		if (edgeStrength[i] >= threshold) {
			mask[i] = 1;
			hasSeed = true;
		}
	}
	if (!hasSeed && edgeStrength.length > 0)
		mask[Math.floor(edgeStrength.length / 2)] = 1;
	return mask;
}

function jumpFloodDistanceTransform(seedMask, width, height) {
	const totalPixels = width * height;
	if (!seedMask || !seedMask.length) {
		const fill = new Float32Array(totalPixels);
		for (let i = 0; i < totalPixels; i++)
			fill[i] = 1;
		return fill;
	}
	const stride = 3;
	const buffers = [
		new Float32Array(totalPixels * stride),
		new Float32Array(totalPixels * stride)
	];
	for (let i = 0; i < totalPixels; i++) {
		const idx = i * stride;
		if (seedMask[i]) {
			const x = i % width;
			const y = Math.floor(i / width);
			buffers[0][idx] = x;
			buffers[0][idx + 1] = y;
			buffers[0][idx + 2] = 0;
		}
		else {
			buffers[0][idx] = -1;
			buffers[0][idx + 1] = -1;
			buffers[0][idx + 2] = 1e9;
		}
	}
	let currentIndex = 0;
	let step = 1;
	const maxDim = Math.max(width, height);
	while (step < maxDim)
		step <<= 1;
	const directions = [
		[-1, -1], [0, -1], [1, -1],
		[-1, 0], [0, 0], [1, 0],
		[-1, 1], [0, 1], [1, 1]
	];
	while (step >= 1) {
		const src = buffers[currentIndex];
		const dst = buffers[1 - currentIndex];
		for (let y = 0; y < height; y++) {
			for (let x = 0; x < width; x++) {
				const pixelIndex = (x + y * width) * stride;
				let bestX = src[pixelIndex];
				let bestY = src[pixelIndex + 1];
				let bestDist = src[pixelIndex + 2];
				for (let i = 0; i < directions.length; i++) {
					const nx = x + directions[i][0] * step;
					const ny = y + directions[i][1] * step;
					if (nx < 0 || ny < 0 || nx >= width || ny >= height)
						continue;
					const neighborIndex = (nx + ny * width) * stride;
					const seedX = src[neighborIndex];
					const seedY = src[neighborIndex + 1];
					if (seedX < 0 || seedY < 0)
						continue;
					const dist = Math.hypot(seedX - x, seedY - y);
					if (dist < bestDist) {
						bestDist = dist;
						bestX = seedX;
						bestY = seedY;
					}
				}
				dst[pixelIndex] = bestX;
				dst[pixelIndex + 1] = bestY;
				dst[pixelIndex + 2] = bestDist;
			}
		}
		currentIndex = 1 - currentIndex;
		step = Math.floor(step / 2);
	}
	const distances = new Float32Array(totalPixels);
	const maxDistance = Math.sqrt(width * width + height * height);
	const src = buffers[currentIndex];
	for (let i = 0; i < totalPixels; i++) {
		const distanceVal = src[i * stride + 2];
		distances[i] = Math.min(1.0, Math.max(0.0, distanceVal / maxDistance));
	}
	return distances;
}

function edgeStrengthToImageData(edgeStrength, width, height) {
	const data = new Uint8ClampedArray(width * height * 4);
	for (let i = 0; i < edgeStrength.length; i++) {
		const inverted = Math.pow(Math.max(0, 1 - edgeStrength[i]), 0.6);
		const value = Math.round(inverted * 255);
		const offset = i * 4;
		data[offset] = value;
		data[offset + 1] = value;
		data[offset + 2] = value;
		data[offset + 3] = 255;
	}
	return new ImageData(data, width, height);
}

function coherenceToImageData(coherence, width, height) {
	const data = new Uint8ClampedArray(width * height * 4);
	for (let i = 0; i < coherence.length; i++) {
		const value = Math.round(Math.pow(coherence[i], 0.8) * 255);
		const offset = i * 4;
		data[offset] = value;
		data[offset + 1] = value;
		data[offset + 2] = value;
		data[offset + 3] = 255;
	}
	return new ImageData(data, width, height);
}

function distanceFieldToImageData(distanceField, width, height) {
	const data = new Uint8ClampedArray(width * height * 4);
	for (let i = 0; i < distanceField.length; i++) {
		const value = Math.round(Math.pow(distanceField[i], 0.5) * 255);
		const offset = i * 4;
		data[offset] = value;
		data[offset + 1] = value;
		data[offset + 2] = value;
		data[offset + 3] = 255;
	}
	return new ImageData(data, width, height);
}

function floatArrayToImageData(values, width, height, invert, gamma) {
	const data = new Uint8ClampedArray(width * height * 4);
	const power = gamma || 1.0;
	for (let i = 0; i < values.length; i++) {
		let value = Math.max(0, Math.min(1, values[i]));
		value = invert ? (1 - value) : value;
		value = Math.pow(value, power);
		const channel = Math.round(value * 255);
		const offset = i * 4;
		data[offset] = channel;
		data[offset + 1] = channel;
		data[offset + 2] = channel;
		data[offset + 3] = 255;
	}
	return new ImageData(data, width, height);
}

function imageDataToCanvas(imageData) {
	const canvas = document.createElement('canvas');
	canvas.width = imageData.width;
	canvas.height = imageData.height;
	const ctx = canvas.getContext('2d');
	ctx.putImageData(imageData, 0, 0);
	return canvas;
}

function computeGrayscale(imageData) {
	const totalPixels = imageData.width * imageData.height;
	const grayscale = new Float32Array(totalPixels);
	for (let i = 0; i < totalPixels; i++) {
		const offset = i * 4;
		const r = imageData.data[offset];
		const g = imageData.data[offset + 1];
		const b = imageData.data[offset + 2];
		grayscale[i] = (r * 0.299 + g * 0.587 + b * 0.114) / 255;
	}
	return grayscale;
}

function computeSobelMagnitude(grayscale, width, height) {
	const output = new Float32Array(grayscale.length);
	for (let y = 1; y < height - 1; y++) {
		for (let x = 1; x < width - 1; x++) {
			const idx = y * width + x;
			const gx = (
				-1.0 * grayscale[(y - 1) * width + (x - 1)] +
				-2.0 * grayscale[y * width + (x - 1)] +
				-1.0 * grayscale[(y + 1) * width + (x - 1)] +
				1.0 * grayscale[(y - 1) * width + (x + 1)] +
				2.0 * grayscale[y * width + (x + 1)] +
				1.0 * grayscale[(y + 1) * width + (x + 1)]
			);
			const gy = (
				-1.0 * grayscale[(y - 1) * width + (x - 1)] +
				-2.0 * grayscale[(y - 1) * width + x] +
				-1.0 * grayscale[(y - 1) * width + (x + 1)] +
				1.0 * grayscale[(y + 1) * width + (x - 1)] +
				2.0 * grayscale[(y + 1) * width + x] +
				1.0 * grayscale[(y + 1) * width + (x + 1)]
			);
			output[idx] = Math.sqrt(gx * gx + gy * gy);
		}
	}
	return output;
}

function normalizeFloatArray(array) {
	let maxVal = 0;
	for (let i = 0; i < array.length; i++)
		maxVal = Math.max(maxVal, array[i]);
	const normalized = new Float32Array(array.length);
	if (maxVal === 0)
		return normalized;
	for (let i = 0; i < array.length; i++)
		normalized[i] = array[i] / maxVal;
	return normalized;
}

function generateKuramotoCoherence(grayscale, width, height) {
	const coherence = new Float32Array(grayscale.length);
	for (let y = 1; y < height - 1; y++) {
		for (let x = 1; x < width - 1; x++) {
			const idx = y * width + x;
			const phase = Math.sin(grayscale[idx] * Math.PI);
			let diffSum = 0;
			let samples = 0;
			const neighbors = [
				[y - 1, x], [y + 1, x],
				[y, x - 1], [y, x + 1]
			];
			for (let i = 0; i < neighbors.length; i++) {
				const ny = neighbors[i][0];
				const nx = neighbors[i][1];
				const nIdx = ny * width + nx;
				const neighborPhase = Math.sin(grayscale[nIdx] * Math.PI);
				diffSum += Math.abs(phase - neighborPhase);
				samples++;
			}
			const avgDiff = (samples > 0) ? diffSum / samples : 0;
			coherence[idx] = Math.max(0, 1 - avgDiff);
		}
	}
	return coherence;
}

// Auto-domain detection: generates domain_id texture from edges/depth/color
// Returns ImageData with domain IDs in R channel (normalized to [0,1])
function generateAutoDomains(mainImageData, depthImageData, edgeImageData, options = {}) {
	const width = mainImageData.width;
	const height = mainImageData.height;
	const totalPixels = width * height;
	
	// Options with defaults
	const {
		edgeThreshold = 0.25,        // Stop growing at edges above this (lower = more sensitive)
		depthThreshold = 0.08,      // Stop growing at depth jumps above this (lower = more sensitive)
		colorThreshold = 0.12,       // Stop growing at color differences above this (lower = more sensitive)
		minRegionSize = 200,        // Merge regions smaller than this (larger = fewer domains)
		seedSpacing = 30,           // Grid spacing for seed points (smaller = more seeds)
		maxDomains = 8              // Maximum number of domains
	} = options;
	
	// Extract feature arrays
	const grayscale = computeGrayscale(mainImageData);
	const depth = new Float32Array(totalPixels);
	const edges = new Float32Array(totalPixels);
	
	// Extract depth and edge data
	for (let i = 0; i < totalPixels; i++) {
		const offset = i * 4;
		if (depthImageData) {
			depth[i] = depthImageData.data[offset] / 255.0;
		} else {
			depth[i] = grayscale[i]; // Fallback to grayscale
		}
		if (edgeImageData) {
			// Edge maps: typically white = far from edge, black = at edge
			// So we invert: high value = low edge strength
			edges[i] = 1.0 - (edgeImageData.data[offset] / 255.0);
		} else {
			edges[i] = 1.0; // No edges = all low edge strength
		}
	}
	
	// Compute color in Lab-like space (simplified)
	const colorL = new Float32Array(totalPixels);
	const colorA = new Float32Array(totalPixels);
	const colorB = new Float32Array(totalPixels);
	for (let i = 0; i < totalPixels; i++) {
		const offset = i * 4;
		const r = mainImageData.data[offset] / 255.0;
		const g = mainImageData.data[offset + 1] / 255.0;
		const b = mainImageData.data[offset + 2] / 255.0;
		// Simple Lab approximation
		colorL[i] = (r * 0.299 + g * 0.587 + b * 0.114);
		colorA[i] = (r - g) * 0.5 + 0.5;
		colorB[i] = (g - b) * 0.5 + 0.5;
	}
	
	// Initialize domain map
	const domainMap = new Int32Array(totalPixels);
	for (let i = 0; i < totalPixels; i++)
		domainMap[i] = -1; // -1 = unassigned
	
	// Find seed points (low edge strength, evenly spaced)
	const seeds = [];
	for (let y = seedSpacing; y < height - seedSpacing; y += seedSpacing) {
		for (let x = seedSpacing; x < width - seedSpacing; x += seedSpacing) {
			const idx = y * width + x;
			// Prefer seeds in low-edge areas (edges[idx] is edge strength: 0 = far from edge, 1 = at edge)
			// So we want seeds where edge strength is LOW (far from edges)
			if (edges[idx] < edgeThreshold * 0.5) {
				seeds.push({x, y, idx});
			}
		}
	}
	
	// If we don't have enough seeds, add more from anywhere
	if (seeds.length < 10) {
		for (let y = seedSpacing; y < height - seedSpacing; y += seedSpacing) {
			for (let x = seedSpacing; x < width - seedSpacing; x += seedSpacing) {
				const idx = y * width + x;
				if (domainMap[idx] === -1) {
					seeds.push({x, y, idx});
				}
			}
		}
	}
	
	// Region growing from seeds
	let currentDomainId = 0;
	const domainStats = []; // Track stats for each domain
	
	function canGrow(fromIdx, toIdx) {
		// Check edge barrier (edges[idx] is edge strength: 0 = far from edge, 1 = at edge)
		// Don't grow across strong edges (high edge strength)
		if (edges[toIdx] > edgeThreshold)
			return false;
		
		// Check depth discontinuity
		const depthDiff = Math.abs(depth[fromIdx] - depth[toIdx]);
		if (depthDiff > depthThreshold)
			return false;
		
		// Check color similarity
		const colorDiff = Math.sqrt(
			Math.pow(colorL[fromIdx] - colorL[toIdx], 2) +
			Math.pow(colorA[fromIdx] - colorA[toIdx], 2) +
			Math.pow(colorB[fromIdx] - colorB[toIdx], 2)
		);
		if (colorDiff > colorThreshold)
			return false;
		
		return true;
	}
	
	// Grow regions from each seed
	for (const seed of seeds) {
		if (domainMap[seed.idx] !== -1)
			continue; // Already assigned
		
		const domainId = currentDomainId++;
		if (domainId >= maxDomains)
			break;
		
		const queue = [seed.idx];
		const domainPixels = [];
		
		while (queue.length > 0) {
			const idx = queue.shift();
			if (domainMap[idx] !== -1)
				continue;
			
			domainMap[idx] = domainId;
			domainPixels.push(idx);
			
			const x = idx % width;
			const y = Math.floor(idx / width);
			
			// Check 4-connected neighbors
			const neighbors = [
				[y - 1, x], [y + 1, x],
				[y, x - 1], [y, x + 1]
			];
			
			for (const [ny, nx] of neighbors) {
				if (ny < 0 || ny >= height || nx < 0 || nx >= width)
					continue;
				
				const nIdx = ny * width + nx;
				if (domainMap[nIdx] === -1 && canGrow(idx, nIdx)) {
					queue.push(nIdx);
				}
			}
		}
		
		domainStats.push({
			id: domainId,
			size: domainPixels.length,
			meanDepth: domainPixels.reduce((sum, i) => sum + depth[i], 0) / domainPixels.length,
			meanColor: {
				L: domainPixels.reduce((sum, i) => sum + colorL[i], 0) / domainPixels.length,
				A: domainPixels.reduce((sum, i) => sum + colorA[i], 0) / domainPixels.length,
				B: domainPixels.reduce((sum, i) => sum + colorB[i], 0) / domainPixels.length
			}
		});
	}
	
	// Assign remaining unassigned pixels to nearest domain
	for (let i = 0; i < totalPixels; i++) {
		if (domainMap[i] === -1) {
			// Find nearest assigned pixel
			let bestDomain = 0;
			let bestDist = Infinity;
			const x = i % width;
			const y = Math.floor(i / width);
			
			for (let dy = -seedSpacing; dy <= seedSpacing && bestDist > 0; dy++) {
				for (let dx = -seedSpacing; dx <= seedSpacing; dx++) {
					const nx = x + dx;
					const ny = y + dy;
					if (nx < 0 || nx >= width || ny < 0 || ny >= height)
						continue;
					const nIdx = ny * width + nx;
					if (domainMap[nIdx] >= 0) {
						const dist = dx * dx + dy * dy;
						if (dist < bestDist) {
							bestDist = dist;
							bestDomain = domainMap[nIdx];
						}
					}
				}
			}
			domainMap[i] = bestDomain;
		}
	}
	
	// Merge small regions into neighbors
	const regionSizes = new Array(currentDomainId).fill(0);
	for (let i = 0; i < totalPixels; i++) {
		if (domainMap[i] >= 0)
			regionSizes[domainMap[i]]++;
	}
	
	for (let domainId = 0; domainId < currentDomainId; domainId++) {
		if (regionSizes[domainId] < minRegionSize) {
			// Merge into nearest larger region
			let bestNeighbor = -1;
			let bestSimilarity = -1;
			
			for (let otherId = 0; otherId < currentDomainId; otherId++) {
				if (otherId === domainId || regionSizes[otherId] < minRegionSize)
					continue;
				
				const similarity = 1.0 - Math.abs(domainStats[domainId].meanDepth - domainStats[otherId].meanDepth);
				if (similarity > bestSimilarity) {
					bestSimilarity = similarity;
					bestNeighbor = otherId;
				}
			}
			
			if (bestNeighbor >= 0) {
				for (let i = 0; i < totalPixels; i++) {
					if (domainMap[i] === domainId)
						domainMap[i] = bestNeighbor;
				}
			}
		}
	}
	
	// Remap domain IDs to be contiguous (0, 1, 2, ...)
	const idMap = new Map();
	let newId = 0;
	for (let i = 0; i < totalPixels; i++) {
		const oldId = domainMap[i];
		if (!idMap.has(oldId)) {
			idMap.set(oldId, newId++);
		}
		domainMap[i] = idMap.get(oldId);
	}
	
	const numDomains = newId;
	
	// Debug: log region sizes before remapping
	const finalRegionSizes = new Array(numDomains).fill(0);
	for (let i = 0; i < totalPixels; i++) {
		if (domainMap[i] >= 0)
			finalRegionSizes[domainMap[i]]++;
	}
	// Log domain generation results
	if (numDomains > 1) {
		console.log(`Auto-generated ${numDomains} domains from image. Region sizes:`, finalRegionSizes);
	} else {
		console.warn(`Auto-domain detection found only ${numDomains} domain(s). Try adjusting thresholds or check edge/depth maps.`);
	}
	
	// Create ImageData with domain IDs normalized to [0,1] in R channel
	const outputImageData = new ImageData(width, height);
	for (let i = 0; i < totalPixels; i++) {
		const offset = i * 4;
		const domainId = domainMap[i];
		// Normalize domain ID to [0,1] range
		const normalizedId = numDomains > 1 ? domainId / (numDomains - 1) : 0.0;
		outputImageData.data[offset] = Math.round(normalizedId * 255);
		outputImageData.data[offset + 1] = 0; // G unused
		outputImageData.data[offset + 2] = 0; // B unused
		outputImageData.data[offset + 3] = 255; // A = 1.0
	}
	
	return { imageData: outputImageData, numDomains, domainMap };
}


// ---------------------- Vector helpers ----------------------

function adjustWavelengthAndAngleForPerfectWraparound(rectWidth, rectHeight, desiredWavelength, desiredAngleRadians) {
	// console.log("adjustWavelengthAndAngleForPerfectWraparound", rectWidth, rectHeight, desiredWavelength, desiredAngleRadians, desiredAngleRadians * 180 / Math.PI);
	// cooperate with our pacman wraparound aka torus topology
	// to get a continuous transition across the edges, we need to round to the nearest whole number of waves
	// yes, this unfortunately means that you can't get exactly the wavelength and angle you asked for
	const numWholeWavesFitX = Math.round(rectWidth * Math.cos(desiredAngleRadians) / desiredWavelength);
	const numWholeWavesFitY = Math.round(rectHeight * Math.sin(desiredAngleRadians) / desiredWavelength);
	// console.log(numWholeWavesFitX, numWholeWavesFitY);
	console.assert(numWholeWavesFitX != 0 || numWholeWavesFitY != 0);
	const x = rectWidth / numWholeWavesFitX;
	const y = rectHeight / numWholeWavesFitY;
	// console.log(x, y);
	// x and y are not usual components of a usual vector, they are somehow the "inverse"
	// we have an infinite line passing through (x,0) and (0,y)
	// which point P on that line is closest to (0,0)?
	// i have a hunch there exists some shorter way to calculate this, but not sure how
	let P;
	if (numWholeWavesFitX == 0) { // the waves are completely vertical
		P = vec2(0, y);
	}
	else if (numWholeWavesFitY == 0) { // the waves are completely horizontal
		P = vec2(x, 0);
	}
	else {
		const lineStart = vec2(x,0);
		const lineEnd = vec2(0,y);
		const a = vec2Difference(vec2(0,0), lineStart);
		const b = vec2Difference(lineEnd, lineStart);
		const position01OfProjectedDotAlongLine = (dot(a,b) / dot(b,b));
		// console.log(position01OfProjectedDotAlongLine);
		// "position01OfProjectedDotAlongLine" indicates the point's position along the line,
		// such that 0 means the point is exactly at lineStart, and 1 means the point is exactly at lineEnd
		// The point P can then be found:
		P = vec2Add(lineStart, vec2MulScalar(b, position01OfProjectedDotAlongLine));
	}
	// console.log(P);
	// get the distance and angle to P
	const adjustedWavelength = vec2Magnitude(P);
	// const adjustedAngle = Math.atan2(P.y, P.x);
	// console.log(adjustedWavelength, adjustedAngle, adjustedAngle * 180 / Math.PI);
	// return [adjustedWavelength, adjustedAngle];
	// actually let's instead return something more tailored to what we need at the single call site
	const inverseLengthP = vec2MulScalar(P, 1/(adjustedWavelength*adjustedWavelength));
	return inverseLengthP;
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
	
	// Check for vertex shader compilation errors
	if (!gl.getShaderParameter(vertexShader, gl.COMPILE_STATUS)) {
		const vertErrLog = gl.getShaderInfoLog(vertexShader);
		setErrorMessage(`Vertex shader ${vertFilename} compilation failed. Error log: ${vertErrLog}`);
		gl.deleteShader(vertexShader);
		return;
	}

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
	
	// Check for fragment shader compilation errors
	if (!gl.getShaderParameter(fragmentShader, gl.COMPILE_STATUS)) {
		const fragErrLog = gl.getShaderInfoLog(fragmentShader);
		setErrorMessage(`Fragment shader ${fragFilename} compilation failed. Error log: ${fragErrLog}`);
		gl.deleteShader(fragmentShader);
		return;
	}

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

function changeFrequenciesAndKernelShrinkageButKeepPhases(gl, iField) {
	if (!g_initOscillatorsShader || !g_initOscillatorsShader.uniformLocations) {
		console.error("g_initOscillatorsShader is not initialized");
		return;
	}

	const confield = structureFieldParamsForShader(iField);
	const oscfield = g_oscillatorFields[iField];

	// console.log("changeFrequenciesAndKernelShrinkageButKeepPhases", confield);

	// a complication is that the oscillator texture might no longer have the same size as when we first created it
	const scaleFactorInActualCurrentUse = g_webglContext.canvas.width / oscfield.oscillatorTextureWidth;
	const osciWidth = g_webglContext.canvas.width / scaleFactorInActualCurrentUse;
	const osciHeight = g_webglContext.canvas.height / scaleFactorInActualCurrentUse;

	gl.useProgram(g_initOscillatorsShader);

	const oscillatorTexRead = oscfield.oscillatorTextures[oscfield.latestIndex];
	gl.activeTexture(gl.TEXTURE0);
	gl.bindTexture(gl.TEXTURE_2D, oscillatorTexRead);
	gl.uniform1i(g_initOscillatorsShader.uniformLocations['uOscillatorTexRead'], 0);

	const oscillatorTexWrite = oscfield.oscillatorTextures[1-oscfield.latestIndex];
	gl.activeTexture(gl.TEXTURE1);
	gl.bindTexture(gl.TEXTURE_2D, oscillatorTexWrite);
	gl.bindFramebuffer(gl.FRAMEBUFFER, g_oscillatorFramebuffer);
	gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, oscillatorTexWrite, 0);

	gl.activeTexture(gl.TEXTURE2);
	gl.bindTexture(gl.TEXTURE_2D, g_mainImageTexture);
	gl.uniform1i(g_initOscillatorsShader.uniformLocations['uMainTex'], 2);

	gl.activeTexture(gl.TEXTURE3);
	gl.bindTexture(gl.TEXTURE_2D, g_mainImageDepthTexture);
	gl.uniform1i(g_initOscillatorsShader.uniformLocations['uDepthTex'], 3);

	gl.activeTexture(gl.TEXTURE4);
	gl.bindTexture(gl.TEXTURE_2D, g_mainImageEdgeTexture);
	gl.uniform1i(g_initOscillatorsShader.uniformLocations['uEdgeTex'], 4);

	gl.uniform1fv(g_initOscillatorsShader.uniformLocations["frequencyRange"], confield.frequencyRange); // note that we do NOT pass in g_frequencySlidersMinMax here
	gl.uniform1i(g_initOscillatorsShader.uniformLocations["frequenciesVaryBy"], confield.frequenciesVaryByInt);
	gl.uniform1f(g_initOscillatorsShader.uniformLocations["frequencyCorrelationPatternSize"], confield.frequencyCorrelationPatternSize / scaleFactorInActualCurrentUse);
	gl.uniform1i(g_initOscillatorsShader.uniformLocations["kernelSizeShrunkBy"], confield.kernelSizeShrunkByInt);
	gl.uniform1i(g_initOscillatorsShader.uniformLocations["logPolarTransform"], confield.logPolar);
	gl.uniform1f(g_initOscillatorsShader.uniformLocations["kernelMaxShrinkDivisor"], 1.0 / confield.kernelShrinkFactor);

	gl.viewport(0, 0, osciWidth, osciHeight);
	renderFullScreen(gl, g_initOscillatorsShader);
	gl.bindFramebuffer(gl.FRAMEBUFFER, null);
	gl.viewport(0, 0, g_webglContext.canvas.width, g_webglContext.canvas.height);

	// swap which one is the latest, for next time
	oscfield.latestIndex = 1 - oscfield.latestIndex;
}

function structureFieldParamsForShader(iField) {
	if (g_params["layer"+iField+"frequenciesVaryBy"] == "depth" && !g_mainImageDepthTexture) {
		console.warn('frequenciesVaryBy == "depth" && !g_mainImageDepthTexture');
		g_params["layer"+iField+"frequenciesVaryBy"] = "random"; // #TODO: also update the html input selection, and disable the invalid one?
	}
	if (g_params["layer"+iField+"frequenciesVaryBy"] == "edge" && !g_mainImageEdgeTexture) {
		console.warn('frequenciesVaryBy == "edge" && !g_mainImageEdgeTexture');
		g_params["layer"+iField+"frequenciesVaryBy"] = "random"; // #TODO: also update the html input selection, and disable the invalid one?
	}
	if (g_params["layer"+iField+"kernelSizeShrunkBy"] == "depth" && !g_mainImageDepthTexture) {
		console.warn('kernelSizeShrunkBy == "depth" && !g_mainImageDepthTexture');
		g_params["layer"+iField+"kernelSizeShrunkBy"] = "nothing"; // #TODO: also update the html input selection, and disable the invalid one?
	}
	if (g_params["layer"+iField+"kernelSizeShrunkBy"] == "edge" && !g_mainImageEdgeTexture) {
		console.warn('kernelSizeShrunkBy == "edge" && !g_mainImageEdgeTexture');
		g_params["layer"+iField+"kernelSizeShrunkBy"] = "nothing"; // #TODO: also update the html input selection, and disable the invalid one?
	}

	const field = {
		kernelMaxDistance: parseFloat(g_params["layer"+iField+"kernelMaxDistance"]),
		fieldResolutionHalvings: parseFloat(g_params["layer"+iField+"fieldResolutionHalvings"]),
		frequencyCorrelationPatternSize: parseFloat(g_params["layer"+iField+"frequencyCorrelationPatternSize"]),
		logPolar: (g_params["layer"+iField+"logPolar"] ? 1 : 0),
		kernelShrinkFactor: parseFloat(g_params["layer"+iField+"kernelShrinkFactor"]),
		kernelRingCoupling: [],
		kernelRingDistances: [],
		kernelRingWidths: [],
		couplingToOtherField: 0,
		frequencyRange: [
			parseFloat(g_params["layer"+iField+"oscillatorFrequencyMin"]),
			parseFloat(g_params["layer"+iField+"oscillatorFrequencyMax"])
		]
	};

	if (g_params.numOscillatorFields == 2)
		field.couplingToOtherField = parseFloat(g_params["layer"+iField+"influenceFromOtherLayer"]);

	for (let iRing = 0; iRing < 4; iRing++) {
		field.kernelRingCoupling[iRing] = parseFloat(g_params["layer"+iField+"ring"+iRing+"coupling"]);
		field.kernelRingDistances[iRing] = parseFloat(g_params["layer"+iField+"ring"+iRing+"distance"]);
		field.kernelRingWidths[iRing] = parseFloat(g_params["layer"+iField+"ring"+iRing+"width"]);
	}
	field.kernelSizeShrunkByInt = 0;
	if (g_params["layer"+iField+"kernelSizeShrunkBy"] == "depth") field.kernelSizeShrunkByInt = 1;
	if (g_params["layer"+iField+"kernelSizeShrunkBy"] == "edge") field.kernelSizeShrunkByInt = 2;
	if (g_params["layer"+iField+"kernelSizeShrunkBy"] == "nothing") field.kernelSizeShrunkByInt = 0;

	field.frequenciesVaryByInt = 0;
	if (g_params["layer"+iField+"frequenciesVaryBy"] == "depth") field.frequenciesVaryByInt = 1;
	if (g_params["layer"+iField+"frequenciesVaryBy"] == "edge") field.frequenciesVaryByInt = 2;
	if (g_params["layer"+iField+"frequenciesVaryBy"] == "brightness") field.frequenciesVaryByInt = 3;
	if (g_params["layer"+iField+"frequenciesVaryBy"] == "pattern") field.frequenciesVaryByInt = 4;
	if (g_params["layer"+iField+"frequenciesVaryBy"] == "random") field.frequenciesVaryByInt = 5;

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
	}


	console.assert(g_params.numOscillatorFields <= g_oscillatorFields.length);

	// g_params contains everything, but not in the right format for passing into shaders
	// so convert it to the right format here before drawing each frame

	const fieldParamsForShader = [];
	for (let iField = 0; iField < g_oscillatorFields.length; iField++)
		fieldParamsForShader[iField] = structureFieldParamsForShader(iField);

	const csdIntensity = updateCorticalDepressionState(timestamp);

	const logPolarTransformLayer = [
		g_params.layer0logPolar,
		g_params.layer1logPolar,
	];

	// first, check if we need to scale up/down the simulation texture for performance reasons
	// this won't happen often, only when the user adjusts the kernel size with the slider
	// don't do the actual resize unless we're going to simulate
	if ((g_isPlaying || g_simulateSingleStep) && !g_debugDrawFrequencies) {
		for (let i = 0; i < g_oscillatorFields.length; i++) {
			let [scaleFactor, scaledKernelMaxDistance, newOscillatorTextureWidth, newOscillatorTextureHeight] = calculateScalingForPerformance(i);
			// console.log('scaleFactor ' + scaleFactor);
			if (g_oscillatorFields[i].oscillatorTextureWidth != newOscillatorTextureWidth) {

				// calculate new size
				g_oscillatorFields[i].oscillatorTextureWidth = newOscillatorTextureWidth;
				console.log('scale oscillator texture '+i+' to ' + newOscillatorTextureWidth + ' * ' + newOscillatorTextureHeight);

				// delete one of the textures and create a new smaller or bigger one
				gl.deleteTexture(g_oscillatorFields[i].oscillatorTextures[1-g_oscillatorFields[i].latestIndex]);
				g_oscillatorFields[i].oscillatorTextures[1-g_oscillatorFields[i].latestIndex] = createOscillatorTexture(g_webglContext, newOscillatorTextureWidth, newOscillatorTextureHeight, null);

				// using a simple shader, draw the texture onto a new scaled texture
				gl.useProgram(g_scalingShader);

				const oscillatorTexRead = g_oscillatorFields[i].oscillatorTextures[g_oscillatorFields[i].latestIndex];
				gl.activeTexture(gl.TEXTURE0);
				gl.bindTexture(gl.TEXTURE_2D, oscillatorTexRead);
				gl.uniform1i(g_scalingShader.uniformLocations['uTexture'], 0);

				const oscillatorTexWrite = g_oscillatorFields[i].oscillatorTextures[1-g_oscillatorFields[i].latestIndex];
				gl.activeTexture(gl.TEXTURE1);
				gl.bindTexture(gl.TEXTURE_2D, oscillatorTexWrite);
				gl.bindFramebuffer(gl.FRAMEBUFFER, g_oscillatorFramebuffer);
				gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, oscillatorTexWrite, 0);

				gl.viewport(0, 0, newOscillatorTextureWidth, newOscillatorTextureHeight);
				renderFullScreen(gl, g_scalingShader);
				gl.bindFramebuffer(gl.FRAMEBUFFER, null);
				gl.viewport(0, 0, g_webglContext.canvas.width, g_webglContext.canvas.height);

				// delete the other one of the textures and create a new smaller or bigger one
				gl.deleteTexture(g_oscillatorFields[i].oscillatorTextures[g_oscillatorFields[i].latestIndex]);
				g_oscillatorFields[i].oscillatorTextures[g_oscillatorFields[i].latestIndex] = createOscillatorTexture(g_webglContext, newOscillatorTextureWidth, newOscillatorTextureHeight, null);

				// swap which one is the latest, for next time
				g_oscillatorFields[i].latestIndex = 1 - g_oscillatorFields[i].latestIndex;
			}
		}


		// secondly, we do the coupled oscillators simulation with a special shader
		gl.useProgram(g_oscillationShader);
		if (g_oscillationShader.uniformLocations["corticalDepressionIntensity"] !== null && g_oscillationShader.uniformLocations["corticalDepressionIntensity"] !== undefined)
			gl.uniform1f(g_oscillationShader.uniformLocations["corticalDepressionIntensity"], csdIntensity);
		// for (let iField in g_oscillatorFields) {
		for (let iField = 0; iField < g_params.numOscillatorFields; iField++) {
			let [scaleFactor, scaledKernelMaxDistance] = calculateScalingForPerformance(iField);

			// read from the latest
			{
				const oscillatorTexRead = g_oscillatorFields[iField].oscillatorTextures[g_oscillatorFields[iField].latestIndex];
				gl.activeTexture(gl.TEXTURE0);
				gl.bindTexture(gl.TEXTURE_2D, oscillatorTexRead);
				gl.uniform1i(g_oscillationShader.uniformLocations['uOscillatorTexRead'], 0);
			}
			// write to the not-latest one
			{
				const oscillatorTexWrite = g_oscillatorFields[iField].oscillatorTextures[1-g_oscillatorFields[iField].latestIndex];
				gl.activeTexture(gl.TEXTURE1);
				gl.bindTexture(gl.TEXTURE_2D, oscillatorTexWrite);
				gl.bindFramebuffer(gl.FRAMEBUFFER, g_oscillatorFramebuffer);
				gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, oscillatorTexWrite, 0);
			}

			// also read from the latest _other_ field, for cross-field coupling
			if (g_params.numOscillatorFields == 2)
			{
				const oscillatorTexRead = g_oscillatorFields[1-iField].oscillatorTextures[g_oscillatorFields[1-iField].latestIndex];
				gl.activeTexture(gl.TEXTURE2);
				gl.bindTexture(gl.TEXTURE_2D, oscillatorTexRead);
				gl.uniform1i(g_oscillationShader.uniformLocations['uOscillatorTexOther'], 2);
				// console.log(couplingToOtherField);
			}
			gl.uniform1f(g_oscillationShader.uniformLocations["couplingToOtherField"], fieldParamsForShader[iField].couplingToOtherField);

			gl.uniform1f(g_oscillationShader.uniformLocations["kernelMaxDistance"], scaledKernelMaxDistance);
			// gl.uniform1f(g_oscillationShader.uniformLocations["couplingSmallWorld"], couplingSmallWorld);
			gl.uniform4fv(g_oscillationShader.uniformLocations["kernelRingCoupling"], fieldParamsForShader[iField].kernelRingCoupling);
			gl.uniform4fv(g_oscillationShader.uniformLocations["kernelRingDistances"], fieldParamsForShader[iField].kernelRingDistances);
			gl.uniform4fv(g_oscillationShader.uniformLocations["kernelRingWidths"], fieldParamsForShader[iField].kernelRingWidths);

			// gl.uniform1i(g_oscillationShader.uniformLocations["debugDrawFilterKernel"], false);
			gl.uniform1f(g_oscillationShader.uniformLocations["deltaTime"], 1 / g_estimatedFps);
			gl.uniform1i(g_oscillationShader.uniformLocations["logPolarTransformMe"], logPolarTransformLayer[iField]);
			gl.uniform1i(g_oscillationShader.uniformLocations["logPolarTransformOther"], logPolarTransformLayer[1-iField]);
			
			// Set domain-related uniforms
			const activeDomainCount = g_domainMaskCPUData ? Math.min(8, g_domainMaskCPUData.numDomains) : 0;
			const useDomains = activeDomainCount > 0 && g_domainMaskTexture;
			const numDomains = useDomains ? activeDomainCount : 0;
			
			gl.uniform1i(g_oscillationShader.uniformLocations["useDomains"], useDomains ? 1 : 0);
			gl.uniform1i(g_oscillationShader.uniformLocations["numDomains"], numDomains);
			
			if (useDomains && g_domainMaskTexture) {
				gl.activeTexture(gl.TEXTURE3);
				gl.bindTexture(gl.TEXTURE_2D, g_domainMaskTexture);
				gl.uniform1i(g_oscillationShader.uniformLocations["uDomainMaskTex"], 3);
				
				const orderLocation = g_oscillationShader.uniformLocations["domainOrderParameter"];
				if (orderLocation !== null && orderLocation !== undefined)
					gl.uniform1fv(orderLocation, g_domainOrderParameter);
				const chaosLocation = g_oscillationShader.uniformLocations["domainChaosParameter"];
				if (chaosLocation !== null && chaosLocation !== undefined)
					gl.uniform1fv(chaosLocation, g_domainChaosParameter);
				
				// Set domain parameters (up to 8 domains)
				const fieldParams = fieldParamsForShader[iField];
				const floatArrayTmp = [];
				for (let i = 0; i < numDomains; i++) {
					let coupling, distances, widths, freqMin, freqMax;

					const baseCoupling = fieldParams ? fieldParams.kernelRingCoupling : [0,0,0,0];
					const baseDistances = fieldParams ? fieldParams.kernelRingDistances : [0.1, 0.3, 0.6, 0.9];
					const baseWidths = fieldParams ? fieldParams.kernelRingWidths : [0.05, 0.08, 0.12, 0.15];
					const baseFreqMin = fieldParams ? fieldParams.frequencyRange[0] : 3.0;
					const baseFreqMax = fieldParams ? fieldParams.frequencyRange[1] : 7.0;
					
					if (g_useAutoDomains && g_autoDomainData && g_autoDomainData.numDomains > 0 && i < g_autoDomainData.numDomains) {
						// Use default parameters for auto-domains (can be customized later)
						// For now, vary parameters slightly based on domain index to create visual distinction
						const domainVariation = numDomains > 1 ? (i / (numDomains - 1)) : 0.0;
						floatArrayTmp.length = 0;
						const couplingScale = 0.85 + (1.15 - 0.85) * domainVariation;
						for (let ring = 0; ring < 4; ring++)
							floatArrayTmp[ring] = (baseCoupling[ring] || 0.0) * couplingScale;
						coupling = new Float32Array(floatArrayTmp);
						floatArrayTmp.length = 0;
						for (let ring = 0; ring < 4; ring++)
							floatArrayTmp[ring] = (baseDistances[ring] !== undefined ? baseDistances[ring] : 0.1 + 0.2 * ring);
						distances = new Float32Array(floatArrayTmp);
						floatArrayTmp.length = 0;
						for (let ring = 0; ring < 4; ring++)
							floatArrayTmp[ring] = (baseWidths[ring] !== undefined ? baseWidths[ring] : 0.05 + 0.03 * ring);
						widths = new Float32Array(floatArrayTmp);
						freqMin = baseFreqMin * (0.8 + 0.4 * domainVariation);
						freqMax = baseFreqMax * (0.8 + 0.4 * domainVariation);
					} else if (g_domains.length > 0 && i < g_domains.length) {
						// Use manual domain parameters
						const domain = g_domains[i];
						coupling = new Float32Array(domain.params.kernelRingCoupling || [0,0,0,0]);
						distances = new Float32Array(domain.params.kernelRingDistances || [0.1,0.3,0.6,0.9]);
						widths = new Float32Array(domain.params.kernelRingWidths || [0.05,0.08,0.12,0.15]);
						freqMin = domain.params.oscillatorFrequencyMin || 3.0;
						freqMax = domain.params.oscillatorFrequencyMax || 7.0;
					} else {
						// Fallback defaults
						coupling = new Float32Array(baseCoupling);
						distances = new Float32Array(baseDistances);
						widths = new Float32Array(baseWidths);
						freqMin = baseFreqMin;
						freqMax = baseFreqMax;
					}
					
					const couplingLocation = g_oscillationShader.uniformLocations[`domainKernelRingCoupling[${i}]`];
					if (couplingLocation !== null && couplingLocation !== undefined)
						gl.uniform4fv(couplingLocation, coupling);
					const distancesLocation = g_oscillationShader.uniformLocations[`domainKernelRingDistances[${i}]`];
					if (distancesLocation !== null && distancesLocation !== undefined)
						gl.uniform4fv(distancesLocation, distances);
					const widthsLocation = g_oscillationShader.uniformLocations[`domainKernelRingWidths[${i}]`];
					if (widthsLocation !== null && widthsLocation !== undefined)
						gl.uniform4fv(widthsLocation, widths);
					const freqMinLocation = g_oscillationShader.uniformLocations[`domainFrequencyMin[${i}]`];
					if (freqMinLocation !== null && freqMinLocation !== undefined)
						gl.uniform1f(freqMinLocation, freqMin);
					const freqMaxLocation = g_oscillationShader.uniformLocations[`domainFrequencyMax[${i}]`];
					if (freqMaxLocation !== null && freqMaxLocation !== undefined)
						gl.uniform1f(freqMaxLocation, freqMax);
				}
			}

			const oscillatorTextureHeight = g_oscillatorFields[iField].oscillatorTextureWidth * g_webglContext.canvas.height / g_webglContext.canvas.width;
			gl.viewport(0, 0, g_oscillatorFields[iField].oscillatorTextureWidth, oscillatorTextureHeight);
			renderFullScreen(gl, g_oscillationShader);
			gl.bindFramebuffer(gl.FRAMEBUFFER, null);
			gl.viewport(0, 0, g_webglContext.canvas.width, g_webglContext.canvas.height);

		}
		// swap which one is the latest, for next frame
		// don't merge this loop into the previous loop, you'd mess up the cross-field coupling
		for (let iField = 0; iField < g_params.numOscillatorFields; iField++) {
			g_oscillatorFields[iField].latestIndex = 1 - g_oscillatorFields[iField].latestIndex;
		}
	}

	const activeDomainCountForCoherence = g_domainMaskCPUData ? Math.min(8, g_domainMaskCPUData.numDomains) : 0;
	if (activeDomainCountForCoherence > 0)
		updateDomainCoherence(gl);

	{
		// thirdly, we render things to the screen with our main shader

		gl.useProgram(g_mainShader);
		if (g_mainShader.uniformLocations["corticalDepressionIntensity"] !== null && g_mainShader.uniformLocations["corticalDepressionIntensity"] !== undefined)
			gl.uniform1f(g_mainShader.uniformLocations["corticalDepressionIntensity"], csdIntensity);
		
		// Tell WebGL we want to affect texture unit 0
		// console.log("before activeTexture");
		gl.activeTexture(gl.TEXTURE0);
		// console.log("before bindTexture");
		// Bind the texture to texture unit 0
		gl.bindTexture(gl.TEXTURE_2D, g_mainImageTexture);
		// Tell the shader we bound the texture to texture unit 0
		gl.uniform1i(g_mainShader.uniformLocations['uMainTex'], 0);
		// console.log("after uniform");
		
		gl.activeTexture(gl.TEXTURE1);
		if (g_params.useCustomVideo) {
			gl.bindTexture(gl.TEXTURE_2D, g_deepDreamVideoTexture);
			//if (g_isPlaying || g_simulateSingleStep) no, actually do it always, otherwise there is no texture before the 1st simulated frame on one of my computers
				updateVideoTexture(gl, g_deepDreamVideo);
		}
		else {
			gl.bindTexture(gl.TEXTURE_2D, g_deepDreamTextures[g_params.deepDreamLayerIndex]);
		}
		gl.uniform1i(g_mainShader.uniformLocations['uDeepDreamTex'], 1);
		
		// console.log(g_displacementVideo, g_displacementVideoTexture, g_displacementVideo.readyState); // prints 4 which means HTMLMediaElement.HAVE_ENOUGH_DATA
		// https://developer.mozilla.org/en-US/docs/Web/API/HTMLMediaElement/readyState

		// #TODO: figure out why the three calls to updateVideoTexture trigger the warning
		// WebGL warning: tex(Sub)Image[23]D: Resource has no data (yet?). Uploading zeros. 3
		// WebGL warning: drawArraysInstanced: TEXTURE_2D at unit 1 is incomplete: The dimensions of `level_base` are not all positive.
		// WebGL warning: drawArraysInstanced: TEXTURE_2D at unit 2 is incomplete: The dimensions of `level_base` are not all positive.
		// WebGL warning: drawArraysInstanced: TEXTURE_2D at unit 3 is incomplete: The dimensions of `level_base` are not all positive. 

		// The image data you pass to texImage2D can be an img element, a video element, ImageData, and more.
		// Because the image data of the video constantly changes, you will have to update the texture
		// inside the requestAnimationFrame animation loop. This is done via texImage2D.
		gl.activeTexture(gl.TEXTURE2);
		gl.bindTexture(gl.TEXTURE_2D, g_displacementVideoTexture);
		// console.log("before", gl.getError()); // Check for errors before
		//if (g_isPlaying || g_simulateSingleStep) no, actually do it always, otherwise there is no texture before the 1st simulated frame on one of my computers
			updateVideoTexture(gl, g_displacementVideo);
		// console.log("after", gl.getError()); // Check for errors after
		gl.uniform1i(g_mainShader.uniformLocations['uDisplacementMap'], 2);

		gl.activeTexture(gl.TEXTURE3);
		gl.bindTexture(gl.TEXTURE_2D, g_blendingPatternVideoTexture);
		//if (g_isPlaying || g_simulateSingleStep) no, actually do it always, otherwise there is no texture before the 1st simulated frame on one of my computers
			updateVideoTexture(gl, g_blendingPatternVideo);
		gl.uniform1i(g_mainShader.uniformLocations['uBlendingPattern'], 3);
		
		// read from the latest
		const index0 = g_debugDrawFrequencies ? g_debugDrawFrequenciesForField : 0; // a dirty trick so that we don't have to pass in g_debugDrawFrequenciesForField to the shader, instead pass in field 1 as 0
		{
			const textureUnitIndices = [];
			gl.activeTexture(gl.TEXTURE4);
			gl.bindTexture(gl.TEXTURE_2D, g_oscillatorFields[index0].oscillatorTextures[g_oscillatorFields[index0].latestIndex]);
			textureUnitIndices.push(4);
			// #TODO: loop instead of these repetitive if()s?
			if (g_params.numOscillatorFields > 1) {
				gl.activeTexture(gl.TEXTURE5);
				gl.bindTexture(gl.TEXTURE_2D, g_oscillatorFields[1].oscillatorTextures[g_oscillatorFields[1].latestIndex]);
				textureUnitIndices.push(5);
			}
			if (g_params.numOscillatorFields > 2) {
				gl.activeTexture(gl.TEXTURE6);
				gl.bindTexture(gl.TEXTURE_2D, g_oscillatorFields[2].oscillatorTextures[g_oscillatorFields[2].latestIndex]);
				textureUnitIndices.push(6);
			}
			if (g_params.numOscillatorFields > 3) {
				gl.activeTexture(gl.TEXTURE7);
				gl.bindTexture(gl.TEXTURE_2D, g_oscillatorFields[3].oscillatorTextures[g_oscillatorFields[3].latestIndex]);
				textureUnitIndices.push(7);
			}
			gl.uniform1iv(g_mainShader.uniformLocations['uOscillatorTexRead'], textureUnitIndices);
			gl.uniform1i(g_mainShader.uniformLocations['uNumOscillatorTextures'], g_params.numOscillatorFields);
		}

		const useDomainsForMain = activeDomainCountForCoherence > 0 && g_domainMaskTexture;
		if (g_mainShader.uniformLocations['useDomains'] !== null && g_mainShader.uniformLocations['useDomains'] !== undefined)
			gl.uniform1i(g_mainShader.uniformLocations['useDomains'], useDomainsForMain ? 1 : 0);
		if (g_mainShader.uniformLocations['numDomains'] !== null && g_mainShader.uniformLocations['numDomains'] !== undefined)
			gl.uniform1i(g_mainShader.uniformLocations['numDomains'], useDomainsForMain ? activeDomainCountForCoherence : 0);
		if (useDomainsForMain && g_domainMaskTexture && g_mainShader.uniformLocations['uDomainMaskTex'] !== null && g_mainShader.uniformLocations['uDomainMaskTex'] !== undefined) {
			gl.activeTexture(gl.TEXTURE8);
			gl.bindTexture(gl.TEXTURE_2D, g_domainMaskTexture);
			gl.uniform1i(g_mainShader.uniformLocations['uDomainMaskTex'], 8);
		}
		if (g_mainShader.uniformLocations['domainOrderParameter'] !== null && g_mainShader.uniformLocations['domainOrderParameter'] !== undefined)
			gl.uniform1fv(g_mainShader.uniformLocations['domainOrderParameter'], g_domainOrderParameter);
		if (g_mainShader.uniformLocations['domainChaosParameter'] !== null && g_mainShader.uniformLocations['domainChaosParameter'] !== undefined)
			gl.uniform1fv(g_mainShader.uniformLocations['domainChaosParameter'], g_domainChaosParameter);


		// Use single slider value for both horizontal and vertical displacements
		gl.uniform1f(g_mainShader.uniformLocations["driftMaxDisplacementHoriz"], g_params.driftMaxDisplacement);
		gl.uniform1f(g_mainShader.uniformLocations["driftMaxDisplacementVert"], g_params.driftMaxDisplacement);
		// uDisplacementMap should be driftPatternSize pixels big on screen in the end, then tiled
		gl.uniform1f(g_mainShader.uniformLocations["driftPatternScale"], gl.canvas.width / g_params.driftPatternSize);
		gl.uniform1f(g_mainShader.uniformLocations["blendPatternScale"], gl.canvas.width / g_params.blendPatternSize);

		gl.uniform1f(g_mainShader.uniformLocations["deepDreamOpacity"], g_params.deepDreamOpacity);
		gl.uniform1f(g_mainShader.uniformLocations["deepLuminosityBlendOpacity"], g_params.deepLuminosityBlendOpacity);
		gl.uniform1f(g_mainShader.uniformLocations["oscillatorOpacity"], g_params.oscillatorOpacity);
		let oscillatorColorOptionIndex = 0;
		if (g_params.oscillatorColors == "rainbow")
			oscillatorColorOptionIndex = 0;
		else if (g_params.oscillatorColors == "rainbow2x")
			oscillatorColorOptionIndex = 1;
		else if (g_params.oscillatorColors == "green-magenta-rota")
			oscillatorColorOptionIndex = 2;
		else if (g_params.oscillatorColors == "green-magenta-sine")
			oscillatorColorOptionIndex = 3;
		else if (g_params.oscillatorColors == "green-magenta-sharp")
			oscillatorColorOptionIndex = 4;
		// else if (g_params.oscillatorColors == "blue-yellow-sharp")
		// 	oscillatorColorOptionIndex = 5;
		else if (g_params.oscillatorColors == "black-white-soft-normal")
			oscillatorColorOptionIndex = 6;
		else if (g_params.oscillatorColors == "black-white-soft-multi")
			oscillatorColorOptionIndex = 7;
		else if (g_params.oscillatorColors == "black-white-sharp-multi")
			oscillatorColorOptionIndex = 8;
		gl.uniform1i(g_mainShader.uniformLocations["oscillatorColors"], oscillatorColorOptionIndex);
		gl.uniform1f(g_mainShader.uniformLocations["crossFieldBlendingBalance"], parseFloat(g_params.crossFieldBlendingBalance)-1);

		gl.uniform1i(g_mainShader.uniformLocations["debugDrawKernelSize"], g_debugDrawFilterKernelShrinkage);
		gl.uniform1i(g_mainShader.uniformLocations["debugDrawKernelSizeForField"], g_debugDrawFilterKernelForField);
		{
			const scaleFactorInActualCurrentUse = gl.canvas.width / g_oscillatorFields[g_debugDrawFilterKernelForField].oscillatorTextureWidth;
			gl.uniform1f(g_mainShader.uniformLocations["kernelMaxDistance"], g_params["layer"+g_debugDrawFilterKernelForField+"kernelMaxDistance"] / scaleFactorInActualCurrentUse);
		}
		gl.uniform1i(g_mainShader.uniformLocations["debugDrawFrequency"], g_debugDrawFrequencies);
		gl.uniform1i(g_mainShader.uniformLocations["frameIndex"], g_debugDrawFrequenciesForNFrames - g_debugDrawFrequencies);
		gl.uniform1fv(g_mainShader.uniformLocations["frequencyRange"], g_frequencySlidersMinMax); // note that we do NOT pass in fieldParamsForShader[iField].frequencyRange here
		gl.uniform1f(g_mainShader.uniformLocations["deltaTime"], 1 / g_estimatedFps);

		if (index0 == 1)
			logPolarTransformLayer[0] = logPolarTransformLayer[1]; // a dirty trick so that we don't have to pass in g_debugDrawFrequenciesForField to the shader, instead pass in field 1 as 0
		gl.uniform1iv(g_mainShader.uniformLocations["logPolarTransform"], logPolarTransformLayer);


		// const oscillatorTextureWidth = g_oscillatorFields[0].oscillatorTextureWidth;
		// const oscillatorTextureHeight = oscillatorTextureWidth * g_webglContext.canvas.height / g_webglContext.canvas.width;
		// gl.viewport(0, g_webglContext.canvas.height - oscillatorTextureHeight, oscillatorTextureWidth, oscillatorTextureHeight);
		renderFullScreen(gl, g_mainShader);
		// gl.viewport(0, 0, g_webglContext.canvas.width, g_webglContext.canvas.height);
	}

	const shouldRequestAnimationFrameAgain = (g_isPlaying || g_debugDrawFrequencies);

	if (g_debugDrawFilterKernel) {
		// fourthly, we draw the ring filter kernel
		gl.useProgram(g_drawKernelShader);

		let [scaleFactor, scaledKernelMaxDistance, newOscillatorTextureWidth, newOscillatorTextureHeight] = calculateScalingForPerformance(g_debugDrawFilterKernelForField);
		const oscillatorTexRead = g_oscillatorFields[g_debugDrawFilterKernelForField].oscillatorTextures[g_oscillatorFields[g_debugDrawFilterKernelForField].latestIndex];

		// pass in the size that the texture will/should have according to the scaleFactor,
		// which is not necessarily the same size as the texture actually has right now, if we haven't resized it yet
		gl.uniform2iv(g_drawKernelShader.uniformLocations['uOscillatorTexSize'], [newOscillatorTextureWidth, newOscillatorTextureHeight]);

		// gl.uniform1f(g_drawKernelShader.uniformLocations["couplingSmallWorld"], couplingSmallWorld);
		gl.uniform1f(g_drawKernelShader.uniformLocations["kernelMaxDistance"], scaledKernelMaxDistance);
		gl.uniform4fv(g_drawKernelShader.uniformLocations["kernelRingCoupling"], fieldParamsForShader[g_debugDrawFilterKernelForField].kernelRingCoupling);
		gl.uniform4fv(g_drawKernelShader.uniformLocations["kernelRingDistances"], fieldParamsForShader[g_debugDrawFilterKernelForField].kernelRingDistances);
		gl.uniform4fv(g_drawKernelShader.uniformLocations["kernelRingWidths"], fieldParamsForShader[g_debugDrawFilterKernelForField].kernelRingWidths);

		// gl.uniform1i(g_drawKernelShader.uniformLocations["debugDrawFilterKernel"], true);

		renderFullScreen(gl, g_drawKernelShader);

		// timer for the kernel preview so it's not visible indefinitely
		if (shouldRequestAnimationFrameAgain)
			g_debugDrawFilterKernel--;
		else // when paused, just draw it for 1 frame
			g_debugDrawFilterKernel = 0;
	}

	if (g_debugDrawFilterKernelShrinkage) {
		if (shouldRequestAnimationFrameAgain)
			g_debugDrawFilterKernelShrinkage--;
		else // when paused, just draw it for 1 frame
			g_debugDrawFilterKernelShrinkage = 0;
	}

	if (g_simulateSingleStep)
		console.log("end single step");

	g_simulateSingleStep = false;
	if (shouldRequestAnimationFrameAgain)
		g_animFrameId = window.requestAnimationFrame(drawWebGl);
	if (g_debugDrawFrequencies)
		g_debugDrawFrequencies--;
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

function calculateScalingForPerformance(iField) {
	let scaleFactor = 1;
	let scaledKernelMaxDistance = g_params["layer"+iField+"kernelMaxDistance"];
	// while (scaledKernelMaxDistance > g_params.kernelLimitForPerformance) {
	for (let i = 0; i < g_params["layer"+iField+"fieldResolutionHalvings"]; i++) {
		scaleFactor *= 2;
		scaledKernelMaxDistance /= 2;
	}
	return [scaleFactor, scaledKernelMaxDistance, Math.round(g_webglContext.canvas.width / scaleFactor), Math.round(g_webglContext.canvas.height / scaleFactor)];
}


// NOTE: these ring kernel functions are implemented in both javascript and glsl shader oscillate.frag,
// so if you change one, remember to change the other to match!

function ringFunction(distance, center, width) {
	const innerGaussian = 0.0;//Math.exp(-Math.pow(distance - center, 2) / (2 * Math.pow(width * 0.5, 2)));
	const outerGaussian = Math.exp(-Math.pow(distance - center, 2) / (2 * Math.pow(width, 2)));
	return innerGaussian - outerGaussian;
}

function influenceFunction(field, distance, maxDistance, couplings) {
	// const couplingSliders = [g_coupling1Slider, g_coupling2Slider, g_coupling3Slider, g_coupling4Slider];
	// const couplings = couplingSliders.map(slider => parseFloat(slider.value));
	let totalInfluence = 0;

	for (let i = 0; i < couplings.length; i++) {
		const r_i = field[iField].kernelRingDistances[i] * maxDistance;
		const width_i = field[iField].kernelRingWidths[i] * maxDistance;
		const A_i = -couplings[i];

		totalInfluence += A_i * ringFunction(distance, r_i, width_i);
	}

	return totalInfluence;
}

function makeConvolutionKernel(field, kernelMaxDistance) {
	const kernel = [];
	for (let dy = -kernelMaxDistance; dy <= kernelMaxDistance; dy++) {
		const kernelRow = [];
		for (let dx = -kernelMaxDistance; dx <= kernelMaxDistance; dx++) {
		    let distance = Math.sqrt(dx * dx + dy * dy);
		    let couplingstrength = influenceFunction(field, distance, kernelMaxDistance, field[iField].kernelRingCoupling);
		    kernelRow.push(couplingstrength);
		}
	    kernel.push(kernelRow);
	}
	return kernel;
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
	return Math.sqrt(a.x * a.x + a.y * a.y);
}

function vec2MulScalar(vec, scalar) {
	return vec2(
		vec.x * scalar,
		vec.y * scalar
	);
}

function dot(vect1, vect2) {
	return vect1.x * vect2.x + vect1.y * vect2.y;
}
