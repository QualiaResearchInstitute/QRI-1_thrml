document.addEventListener('DOMContentLoaded', function() {

	// #TODO: if the URL contains a bunch of parameters, load them from there to the html form input controls
	// if (window.location.search) {
	// 	deserializeConfigStateFromUrl();
	// }

    const toggleSidebarButton = document.getElementById('toggleSidebar');
    const sidebar = document.getElementById('sidebar');
  
    toggleSidebarButton.addEventListener('click', function() {
      sidebar.classList.toggle('d-none');
    });

    const selectPhotoButton = document.getElementById('selectPhotoButton');
    selectPhotoButton.addEventListener('click', function() {
      // The modal is automatically handled by Bootstrap via data-bs-toggle and data-bs-target
      // No additional JavaScript is needed here unless you want to perform actions on open
    });

	// document.querySelectorAll('[data-toggle="tooltip"]').tooltip();
});

/*
// these functions were previously used to store all form input field values / parameters in the URL
// but they were written when all such values were stored in the global object g_configState,
// but now that object is much smaller and contains hardly anything!
// most things are moved to the new global object g_params
// so #TODO: update these functions to work in another new way :)
// maybe in a similar way to exportParametersToJSON

// see also function deserializeConfigStateFromUrl
const g_dummyState = {};
function serializeConfigStateToUrl() {
	const asdf = encodeURI(JSON.stringify(g_params));
	// sigh, dammit, both of these methods create entries in the browser history :(
	// window.location.replace("#" + asdf);
	history.replaceState(g_dummyState, "", "?" + asdf);
}


// see also function serializeConfigStateToUrl
function deserializeConfigStateFromUrl() {
	const serializedConfigState = decodeURI(window.location.search.substr(1));
	// console.log(serializedConfigState);
	const configStateFromUrl = JSON.parse(serializedConfigState);
	// console.log(configStateFromUrl);
	// g_configState = configStateFromUrl; // no, don't do this, gives problems with version compatibility
	// this is slightly better:
	for (let key in g_configState) {
		if (configStateFromUrl.hasOwnProperty(key) && typeof g_configState[key] == typeof configStateFromUrl[key])
			g_configState[key] = configStateFromUrl[key];
	}
}
*/