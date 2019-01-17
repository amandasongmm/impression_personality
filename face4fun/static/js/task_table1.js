/*
 * Requires:
 *     psiturk.js
 *     utils.js
 */


// Initalize psiturk object
var psiTurk = new PsiTurk(uniqueId, adServerLoc, mode);

var mycondition = condition;  // these two variables are passed by the psiturk server process
var mycounterbalance = counterbalance;  // they tell you which condition you have been assigned to
// they are not used in the stroop code but may be useful to you

// All pages to be loaded
var pages = [
	"instructions/instruct-1.html",
	"instructions/instruct-2.html",
	"instructions/instruct-3.html",
	"instructions/instruct-ready.html",
	"stage.html",
	"postquestionnaire.html"
];

psiTurk.preloadPages(pages);

var instructionPages = [ // add as a list as many pages as you like
	"instructions/instruct-1.html",
	"instructions/instruct-2.html",
	"instructions/instruct-3.html",
	"instructions/instruct-ready.html"
];


/********************
* HTML manipulation
*
* All HTML files in the templates directory are requested
* from the server when the PsiTurk object is created above. We
* need code to get those pages from the PsiTurk object and
* insert them into the document.
*
********************/

/********************
* STROOP TEST       *
********************/
var StroopExperiment = function() {

	var wordon, // time word is presented
	    listening = false;

	// Stimuli for a basic Stroop experiment
	// var stims = [
	// 		["SHIP", "red", "unrelated"],
	// 		["MONKEY", "green", "unrelated"],
	// 		["ZAMBONI", "blue", "unrelated"],
	// 		["RED", "red", "congruent"],
	// 		["GREEN", "green", "congruent"],
	// 		["BLUE", "blue", "congruent"],
	// 		["GREEN", "red", "incongruent"],
	// 		["BLUE", "green", "incongruent"],
	// 		["RED", "blue", "incongruent"]
	// 	];

		var stims = [['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/121645.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/088721.jpg', 'ground_truth', '0', '1', 39, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/074137.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/180698.jpg', 'ground_truth', '1', '0', 21, 1],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/032770.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/097670.jpg', 'ground_truth', '0', '1', 2, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/130100.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/140526.jpg', 'ground_truth', '1', '0', 8, 1],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/005407.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/196526.jpg', 'ground_truth', '1', '0', 29, 1],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/106324.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/150710.jpg', 'ground_truth', '1', '0', 36, 1],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/079452.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/006125.jpg', 'ground_truth', '0', '1', 33, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/156437.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/193499.jpg', 'ground_truth', '1', '0', 35, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/089100.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/058683.jpg', 'ground_truth', '0', '1', 9, 1],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/001583.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/022830.jpg', 'ground_truth', '0', '1', 27, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/158098.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/042880.jpg', 'ground_truth', '0', '1', 24, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/197889.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/007582.jpg', 'ground_truth', '0', '1', 12, 1],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/122913.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/043035.jpg', 'ground_truth', '0', '1', 0, 1],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/145078.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/090799.jpg', 'ground_truth', '0', '1', 3, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/124901.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/043904.jpg', 'ground_truth', '1', '0', 26, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/150710.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/106324.jpg', 'ground_truth', '0', '1', 36, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/074137.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/180698.jpg', 'ground_truth', '1', '0', 21, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/152507.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/042311.jpg', 'ground_truth', '0', '1', 10, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/052872.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/059355.jpg', 'ground_truth', '1', '0', 23, 1],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/058863.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/145447.jpg', 'ground_truth', '1', '0', 19, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/150830.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/048555.jpg', 'ground_truth', '0', '1', 14, 1],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/040244.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/097191.jpg', 'ground_truth', '0', '1', 15, 1],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/071470.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/155603.jpg', 'ground_truth', '1', '0', 17, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/034125.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/046885.jpg', 'ground_truth', '0', '1', 31, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/163226.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/073829.jpg', 'ground_truth', '0', '1', 7, 1],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/191219.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/104860.jpg', 'ground_truth', '0', '1', 32, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/001583.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/022830.jpg', 'ground_truth', '0', '1', 27, 1],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/157262.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/065356.jpg', 'ground_truth', '0', '1', 38, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/090799.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/145078.jpg', 'ground_truth', '1', '0', 3, 1],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/128891.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/017814.jpg', 'ground_truth', '0', '1', 13, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/165757.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/051115.jpg', 'ground_truth', '0', '1', 22, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/017814.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/128891.jpg', 'ground_truth', '1', '0', 13, 1],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/028598.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/140728.jpg', 'ground_truth', '0', '1', 30, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/165921.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/049995.jpg', 'ground_truth', '0', '1', 34, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/052872.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/059355.jpg', 'ground_truth', '1', '0', 23, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/024539.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/160268.jpg', 'ground_truth', '1', '0', 4, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/048555.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/150830.jpg', 'ground_truth', '1', '0', 14, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/196526.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/005407.jpg', 'ground_truth', '0', '1', 29, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/172332.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/162314.jpg', 'ground_truth', '0', '1', 28, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/018495.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/014036.jpg', 'ground_truth', '0', '1', 16, 1],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/180662.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/201546.jpg', 'ground_truth', '0', '1', 11, 1],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/069301.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/065764.jpg', 'ground_truth', '1', '0', 18, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/104860.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/191219.jpg', 'ground_truth', '1', '0', 32, 1],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/182776.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/081442.jpg', 'ground_truth', '0', '1', 20, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/020383.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/158343.jpg', 'ground_truth', '1', '0', 25, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/179200.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/116643.jpg', 'ground_truth', '0', '1', 6, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/097670.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/032770.jpg', 'ground_truth', '1', '0', 2, 1],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/073829.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/163226.jpg', 'ground_truth', '1', '0', 7, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/058863.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/145447.jpg', 'ground_truth', '1', '0', 19, 1],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/135214.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/105530.jpg', 'ground_truth', '1', '0', 1, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/018495.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/014036.jpg', 'ground_truth', '0', '1', 16, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/122913.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/043035.jpg', 'ground_truth', '0', '1', 0, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/040244.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/097191.jpg', 'ground_truth', '0', '1', 15, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/151791.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/137263.jpg', 'ground_truth', '1', '0', 5, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/180662.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/201546.jpg', 'ground_truth', '0', '1', 11, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/089100.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/058683.jpg', 'ground_truth', '0', '1', 9, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/140526.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/130100.jpg', 'ground_truth', '0', '1', 8, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/007582.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/197889.jpg', 'ground_truth', '1', '0', 12, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/165757.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/051115.jpg', 'ground_truth', '0', '1', 22, 1],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/156116.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_emotional/125463.jpg', 'ground_truth', '0', '1', 37, 0],];

	stims = _.shuffle(stims);

	var next = function() {
		if (stims.length===0) {
			finish();
		}
		else {
			stim = stims.shift();
			show_word( stim[0], stim[1] );
			wordon = new Date().getTime();
			listening = true;
			d3.select("#query").html('<p id="prompt">Type "A" for left face, "B" for right face.</p>');
		}
	};

	var response_handler = function(e) {
		if (!listening) return;

		var keyCode = e.keyCode,
			response;

		switch (keyCode) {
			case 65:
				// "A"
				response="1";
				break;
			case 66:
				// "B"
				response="0";
				break;
			default:
				response = "";
				break;
		}
		if (response.length>0) {
			listening = false;
			var hit = response == stim[3];
			var rt = new Date().getTime() - wordon;

			psiTurk.recordTrialData({'phase':"TEST",
                                     'im1':stim[0],
                                     'im2':stim[1],
                                     'tasktype':stim[2],
                                     'im1relation':stim[3],
				                     'im2relation':stim[4],
									 'pair_ind':stim[5],
									 'rep': stim[6],
                                     'hit':hit,
                                     'rt':rt}
                                   );
			remove_word();
			next();
		}
	};

	var finish = function() {
	    $("body").unbind("keydown", response_handler); // Unbind keys
	    currentview = new Questionnaire();
	};

	var show_word = function(image1, image2) {
		$('#stim').html('<img src='+image1+' width=50% /><img src='+image2+' width=50% />');

		// d3.select("#stim")
		// 	.append("div")
		// 	.attr("id","word")
		// 	.style("color",color)
		// 	.style("text-align","center")
		// 	.style("font-size","150px")
		// 	.style("font-weight","400")
		// 	.style("margin","20px")
		// 	.text(text);
	};

	var remove_word = function() {
		d3.select("#word").remove();
	};


	// Load the stage.html snippet into the body of the page
	psiTurk.showPage('stage.html');

	// Register the response handler that is defined above to handle any
	// key down events.
	$("body").focus().keydown(response_handler);

	// Start the test
	next();
};


/****************
* Questionnaire *
****************/

var Questionnaire = function() {

	var error_message = "<h1>Oops!</h1><p>Something went wrong submitting your HIT. This might happen if you lose your internet connection. Press the button to resubmit.</p><button id='resubmit'>Resubmit</button>";

	record_responses = function() {

		psiTurk.recordTrialData({'phase':'postquestionnaire', 'status':'submit'});

		$('textarea').each( function(i, val) {
			psiTurk.recordUnstructuredData(this.id, this.value);
		});
		$('select').each( function(i, val) {
			psiTurk.recordUnstructuredData(this.id, this.value);
		});

	};

	prompt_resubmit = function() {
		document.body.innerHTML = error_message;
		$("#resubmit").click(resubmit);
	};

	resubmit = function() {
		document.body.innerHTML = "<h1>Trying to resubmit...</h1>";
		reprompt = setTimeout(prompt_resubmit, 10000);

		psiTurk.saveData({
			success: function() {
			    clearInterval(reprompt);
                psiTurk.computeBonus('compute_bonus', function(){
                	psiTurk.completeHIT(); // when finished saving compute bonus, the quit
                });


			},
			error: prompt_resubmit
		});
	};

	// Load the questionnaire snippet
	psiTurk.showPage('postquestionnaire.html');
	psiTurk.recordTrialData({'phase':'postquestionnaire', 'status':'begin'});

	$("#next").click(function () {
	    record_responses();
	    psiTurk.saveData({
            success: function(){
                psiTurk.computeBonus('compute_bonus', function() {
                	psiTurk.completeHIT(); // when finished saving compute bonus, the quit
                });
            },
            error: prompt_resubmit});
	});


};

// Task object to keep track of the current phase
var currentview;

/*******************
 * Run Task
 ******************/
$(window).load( function(){
    psiTurk.doInstructions(
    	instructionPages, // a list of pages you want to display in sequence
    	function() { currentview = new StroopExperiment(); } // what you want to do when you are done with instructions
    );
});