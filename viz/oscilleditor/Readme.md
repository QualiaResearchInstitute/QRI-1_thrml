# WHAT IS IT

A tool for replicating visual hallucinations. Includes several different effects.
Maybe the most important effect is coupled oscillators (Kuramoto model) on a 2d grid implemented in a WebGL 2 fragment shader.


# HOW TO RUN

Don't just open index.html as a local file from disk, it won't work. (Because of browser security settings that block it from downloading other files via XmlHttpRequest / fetch, which we use for shaders)
You need to run it from a webserver. Any webserver will do. For example, if you have Python installed, you could run this command in this current directory:
	python3 -m http.server
and then surf to http://localhost:8000/

That is a bit inconvenient. We could change it so it works to just open the html file. How? We could put the shader code in the html file instead. With these kind of tags:
	<script type="x-shader/x-fragment" id="fragment-shader">glsl here...</script>
	source = document.querySelector("#fragment-shader").innerHTML;
But it's worse for developers, because then you don't get syntax highlighting when you write the shader code.


# TODO as of 2024-12-09:
* Add helpful tool tips to all the GUI controls.
* When the user changes any GUI control, preview its effect immediately. How exactly to preview it will vary from param to param. Currently these previews are missing or bad:
	* "Drifting speed" and "Blending mask pattern speed" are only shown while playing, not while paused
	* "Drifting pattern size" and "Blending mask pattern size" are hard to see
	* "Coupling to layer" are not visualized at all
* If you choose "Edge map", but the photo doesn't have an associated edge map, warn the user somehow! (not just in browser developer console)
* The "Resolution halvings" slider that's used for performance tweaking is an extra timewasting step when you just want to change the kernel radius, so add a more automatic method instead. What should it be based on? What matters most for performance is the total number of pixels that will be sampled in the innermost loop of the convolution kernel, i e canvaswidth * canvasheight * kernelsize * kernelsize. Maybe change so we scale up/down based on that? And then a slider so the user can tune the pixel limit for their machine.
* And of course: add more effects! More more more!
* Rainbow colors are currently created by a crude method. Change to something more perceptually correct like LAB color.
* Don't download shader.vert 5 times. Download it just once and reuse it.
* For speed optimization, figure out how to use more complex number math in oscillate.frag, rather than converting to and from angles all the time. The 2d cross product should work, why does it not?
