// Google Sheets Integration for Replication App
// This file handles sending form data and parameters to Google Sheets

// Configuration - Update this URL with your deployed Google Apps Script web app URL. See google-apps-script-webapp.js for further instructions
const GOOGLE_SHEETS_WEB_APP_URL = 'https://script.google.com/macros/s/AKfycbydWfivi0o2En4JVxV3_U6Y1fHXJFoJjNAUALWlWxFrDNh4YOIIF1baIbdGolraIls8bg/exec';

/**
 * Shows a status message to the user
 * @param {string} message - Message to display
 * @param {string} type - Type of message (success, error, info)
 */
function showStatusMessage(message, type = 'info') {
	// Create or update status message element
	let statusElement = document.getElementById('submitStatus');
	if (!statusElement) {
		statusElement = document.createElement('div');
		statusElement.id = 'submitStatus';
		statusElement.className = 'alert mt-2';

		// Insert after the submit button
		const modal = document.querySelector('#submitParamsModal .modal-body');
		modal.appendChild(statusElement);
	}

	// Update classes and message
	statusElement.className = `alert mt-2 alert-${type === 'success' ? 'success' : (type === 'error' ? 'danger' : 'info')}`;
	statusElement.textContent = message;
	statusElement.style.display = 'block';

	// Auto-hide after 5 seconds for success messages
	if (type === 'success') {
		setTimeout(() => {
			statusElement.style.display = 'none';
		}, 5000);
	}
}


async function clickSubmitToSpreadsheet() {
	const submitButton = document.querySelector('#submitParamsModal .btn-primary');
	const originalText = submitButton.textContent;

	try {
		// Show loading state
		submitButton.disabled = true;
		submitButton.textContent = 'Submitting...';
		showStatusMessage('Submitting data...', 'info');

		// Collect survey form data
		const form = document.querySelector('#submitParamsModal form');
		const formDataAuto = new FormData(form); // note that radio button only get included if one radio button in the group is checked, otherwise the radio button is completely missing from here
		// console.log(form, formDataAuto, form["drug"], form["inAlteredState"]);
		const formDataNormalObject = {}; // convert because FormData objects apparently can't be json stringified
		for (const pair of formDataAuto)
			if (pair[0] != "sharingUrl")
				formDataNormalObject[pair[0]] = pair[1].trim();

		// Collect browser and environment info
		// no, let's not gather this info, we have no clear use for it and it _could_ be used for fingerprinting = less anonymity
		// const environmentState = {
		// 	userAgent: navigator.userAgent,
		// 	screenResolution: `${screen.width}x${screen.height}`,
		// 	windowSize: `${window.innerWidth}x${window.innerHeight}`,
		// 	pixelRatio: window.devicePixelRatio || 1,
		// 	canvasWidth: document.getElementById('glCanvas')?.width || 0,
		// 	canvasHeight: document.getElementById('glCanvas')?.height || 0,
		// 	currentPhotoFilename: sourcePhotos[g_params.photoIndex].base, // pointless because we already store the photoIndex in the JSON / URL
		// 	timestamp: new Date().toISOString() // pointless because we already have it in payload
		// };

		const payload = {
			formData: formDataNormalObject,
			timestamp: new Date().toISOString(),
			//parameterCount: Object.keys(g_params).length,
			parameters: g_params,
			url: window.location.origin + window.location.pathname + '?' + encodeURI(urlQueryStringFromObject(g_params)),
		};
		console.log(payload, JSON.stringify(payload));

		// Basic validation
		// outcommented until we've decided what questions to ask
		// if (!formData.drug && !formData.inAlteredState)
		// 	throw new Error('Please fill in at least the drug information or altered state selection.');

		console.assert(GOOGLE_SHEETS_WEB_APP_URL);

		let result;
		try {
			// Submit to Google Sheets
			const response = await fetch(GOOGLE_SHEETS_WEB_APP_URL, {
				redirect: "follow",
				method: 'POST',
				headers: {
					'Content-Type': 'text/plain;charset=utf-8',
				},
				body: JSON.stringify(payload)
			});

			if (!response.ok) {
				throw new Error(`HTTP error! status: ${response.status}`);
			}

			result = await response.json();

		} catch (error) {
		    console.error('Error submitting to Google Sheets:', error);
		    throw error;
		}


		if (result.success) {
			showStatusMessage(`Data submitted successfully!`, 'success');
			// console.log(`Row ${result.rowNumber} added to spreadsheet`); // don't leak this to end users

			// close modal after a delay
			setTimeout(() => {
				// Close the modal
				const modal = bootstrap.Modal.getInstance(document.getElementById('submitParamsModal'));
				if (modal)
					modal.hide();

				// Clear form
				document.querySelector('#submitParamsModal form').reset();
			}, 2000);

		}
		else {
			throw new Error(result.message || 'Unknown error occurred');
		}

	}
	catch (error) {
		console.error('Submit error:', error);
		showStatusMessage(`Error: ${error.message}`, 'error');
	}
	finally {
		// Restore button state
		submitButton.disabled = false;
		submitButton.textContent = originalText;
	}
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
	// Add event listener to submit button
	document.querySelector('#submitParamsModal .btn-primary').addEventListener('click', clickSubmitToSpreadsheet);

	// Add event listener to close/cancel buttons to clear status
	const closeButtons = document.querySelectorAll('#submitParamsModal .btn-secondary, #submitParamsModal .btn-close');
	closeButtons.forEach(button => {
		button.addEventListener('click', () => {
			const statusElement = document.getElementById('submitStatus');
			if (statusElement) {
				statusElement.style.display = 'none';
			}
		});
	});

	document.getElementById('submitParamsModal').addEventListener('show.bs.modal', function (e) {
		const sharingUrlText = document.querySelector('#submitParamsModal form input[name="sharingUrl"]');
		sharingUrlText.value = window.location.origin + window.location.pathname + '?' + encodeURI(urlQueryStringFromObject(g_params));
	});
});