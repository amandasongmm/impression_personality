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
		var stims = [
		    ['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087481_0.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087581_9.0_AE.png', 'Different', '0', '1', 85, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087437_9.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087437_0.0_AE.png', 'Same', '1', '0', 44, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087543_9.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087439_0.0_AE.png', 'Different', '1', '0', 49, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087463_0.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087563_9.0_AE.png', 'Different', '0', '1', 68, 1],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087462_0.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087562_9.0_AE.png', 'Different', '0', '1', 67, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087428_9.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087428_0.0_AE.png', 'Same', '1', '0', 36, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087413_0.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087413_9.0_AE.png', 'Same', '0', '1', 21, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087566_9.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087466_0.0_AE.png', 'Different', '1', '0', 71, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087430_0.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087430_9.0_AE.png', 'Same', '0', '1', 38, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087585_9.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087485_0.0_AE.png', 'Different', '1', '0', 88, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087484_0.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087584_9.0_AE.png', 'Different', '0', '1', 87, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087486_0.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087534_9.0_AE.png', 'Different', '0', '1', 89, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/076493.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/101907.jpg', 'ground_truth', '1', '0', 92, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087478_0.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087578_9.0_AE.png', 'Different', '0', '1', 82, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087574_9.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087474_0.0_AE.png', 'Different', '1', '0', 78, 1],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087440_0.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087580_9.0_AE.png', 'Different', '0', '1', 46, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087392_9.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087392_0.0_AE.png', 'Same', '1', '0', 1, 1],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087448_0.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087557_9.0_AE.png', 'Different', '0', '1', 54, 1],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087551_9.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087451_0.0_AE.png', 'Different', '1', '0', 57, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087546_9.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087446_0.0_AE.png', 'Different', '1', '0', 52, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087405_0.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087405_9.0_AE.png', 'Same', '0', '1', 13, 1],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087399_0.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087399_9.0_AE.png', 'Same', '0', '1', 7, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087569_9.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087469_0.0_AE.png', 'Different', '1', '0', 73, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087415_9.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087415_0.0_AE.png', 'Same', '1', '0', 23, 1],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087558_9.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087458_0.0_AE.png', 'Different', '1', '0', 63, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087582_9.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087483_0.0_AE.png', 'Different', '1', '0', 86, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087417_0.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087417_9.0_AE.png', 'Same', '0', '1', 25, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087426_0.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087426_9.0_AE.png', 'Same', '0', '1', 34, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087577_9.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087477_0.0_AE.png', 'Different', '1', '0', 81, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087541_9.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087441_0.0_AE.png', 'Different', '1', '0', 47, 1],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087415_0.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087415_9.0_AE.png', 'Same', '0', '1', 23, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087424_0.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087424_9.0_AE.png', 'Same', '0', '1', 32, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087442_0.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087542_9.0_AE.png', 'Different', '0', '1', 48, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087550_9.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087450_0.0_AE.png', 'Different', '1', '0', 56, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087391_0.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087391_9.0_AE.png', 'Same', '0', '1', 0, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087436_0.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087436_9.0_AE.png', 'Same', '0', '1', 43, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087414_0.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087414_9.0_AE.png', 'Same', '0', '1', 22, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087465_0.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087468_9.0_AE.png', 'Different', '0', '1', 70, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087420_0.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087420_9.0_AE.png', 'Same', '0', '1', 28, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087401_0.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087401_9.0_AE.png', 'Same', '0', '1', 9, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087419_9.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087419_0.0_AE.png', 'Same', '1', '0', 27, 1],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087409_0.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087409_9.0_AE.png', 'Same', '0', '1', 17, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087418_9.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087418_0.0_AE.png', 'Same', '1', '0', 26, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087551_9.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087451_0.0_AE.png', 'Different', '1', '0', 57, 1],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087421_9.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087421_0.0_AE.png', 'Same', '1', '0', 29, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087553_9.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087453_0.0_AE.png', 'Different', '1', '0', 59, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087575_9.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087475_0.0_AE.png', 'Different', '1', '0', 79, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087471_0.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087571_9.0_AE.png', 'Different', '0', '1', 75, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087422_9.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087422_0.0_AE.png', 'Same', '1', '0', 30, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087438_0.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087538_9.0_AE.png', 'Different', '0', '1', 45, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087449_0.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087549_9.0_AE.png', 'Different', '0', '1', 55, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087574_9.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087474_0.0_AE.png', 'Different', '1', '0', 78, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087405_0.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087405_9.0_AE.png', 'Same', '0', '1', 13, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087416_0.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087416_9.0_AE.png', 'Same', '0', '1', 24, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087463_0.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087563_9.0_AE.png', 'Different', '0', '1', 68, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087397_9.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087397_0.0_AE.png', 'Same', '1', '0', 5, 1],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087124.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/038546.jpg', 'ground_truth', '0', '1', 94, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087467_0.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087567_9.0_AE.png', 'Different', '0', '1', 72, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087397_0.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087397_9.0_AE.png', 'Same', '0', '1', 5, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087425_9.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087425_0.0_AE.png', 'Same', '1', '0', 33, 1],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087423_0.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087423_9.0_AE.png', 'Same', '0', '1', 31, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087433_0.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087433_9.0_AE.png', 'Same', '0', '1', 41, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087427_9.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087427_0.0_AE.png', 'Same', '1', '0', 35, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087434_9.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087434_0.0_AE.png', 'Same', '1', '0', 42, 1],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087434_9.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087434_0.0_AE.png', 'Same', '1', '0', 42, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087408_0.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087408_9.0_AE.png', 'Same', '0', '1', 16, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087425_9.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087425_0.0_AE.png', 'Same', '1', '0', 33, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087406_0.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087406_9.0_AE.png', 'Same', '0', '1', 14, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087540_9.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087480_0.0_AE.png', 'Different', '1', '0', 84, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087459_0.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087559_9.0_AE.png', 'Different', '0', '1', 64, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087413_0.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087413_9.0_AE.png', 'Same', '0', '1', 21, 1],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087431_0.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087431_9.0_AE.png', 'Same', '0', '1', 39, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087403_9.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087403_0.0_AE.png', 'Same', '1', '0', 11, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087454_0.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087554_9.0_AE.png', 'Different', '0', '1', 60, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/066743.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/127601.jpg', 'ground_truth', '0', '1', 93, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087447_0.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087547_9.0_AE.png', 'Different', '0', '1', 53, 1],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087412_9.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087412_0.0_AE.png', 'Same', '1', '0', 20, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087554_9.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087454_0.0_AE.png', 'Different', '1', '0', 60, 1],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087429_9.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087429_0.0_AE.png', 'Same', '1', '0', 37, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087476_0.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087568_9.0_AE.png', 'Different', '0', '1', 80, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/095605.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/144125.jpg', 'ground_truth', '0', '1', 98, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087544_9.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087444_0.0_AE.png', 'Different', '1', '0', 50, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/113079.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/086142.jpg', 'ground_truth', '1', '0', 90, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087534_9.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087486_0.0_AE.png', 'Different', '1', '0', 89, 1],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087441_0.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087541_9.0_AE.png', 'Different', '0', '1', 47, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087404_0.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087404_9.0_AE.png', 'Same', '0', '1', 12, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087445_0.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087545_9.0_AE.png', 'Different', '0', '1', 51, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087547_9.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087447_0.0_AE.png', 'Different', '1', '0', 53, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/028757.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/063159.jpg', 'ground_truth', '1', '0', 97, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087461_0.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087561_9.0_AE.png', 'Different', '0', '1', 66, 1],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087394_9.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087394_0.0_AE.png', 'Same', '1', '0', 3, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087472_0.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087572_9.0_AE.png', 'Different', '0', '1', 76, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087479_0.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087579_9.0_AE.png', 'Different', '0', '1', 83, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087473_0.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087573_9.0_AE.png', 'Different', '0', '1', 77, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087452_0.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087552_9.0_AE.png', 'Different', '0', '1', 58, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087450_0.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087550_9.0_AE.png', 'Different', '0', '1', 56, 1],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087396_9.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087396_0.0_AE.png', 'Same', '1', '0', 4, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087411_0.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087411_9.0_AE.png', 'Same', '0', '1', 19, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087410_0.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087410_9.0_AE.png', 'Same', '0', '1', 18, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087392_0.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087392_9.0_AE.png', 'Same', '0', '1', 1, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087570_9.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087470_0.0_AE.png', 'Different', '1', '0', 74, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087400_9.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087400_0.0_AE.png', 'Same', '1', '0', 8, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087542_9.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087442_0.0_AE.png', 'Different', '1', '0', 48, 1],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087402_9.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087402_0.0_AE.png', 'Same', '1', '0', 10, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087432_9.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087432_0.0_AE.png', 'Same', '1', '0', 40, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087407_0.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087407_9.0_AE.png', 'Same', '0', '1', 15, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/029805.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/142336.jpg', 'ground_truth', '0', '1', 96, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087455_0.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087555_9.0_AE.png', 'Different', '0', '1', 61, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087461_0.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087561_9.0_AE.png', 'Different', '0', '1', 66, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/046077.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/065232.jpg', 'ground_truth', '0', '1', 91, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087398_9.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087398_0.0_AE.png', 'Same', '1', '0', 6, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087557_9.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087448_0.0_AE.png', 'Different', '1', '0', 54, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087464_0.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087529_9.0_AE.png', 'Different', '0', '1', 69, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/031763.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/071197.jpg', 'ground_truth', '0', '1', 95, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087460_0.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087560_9.0_AE.png', 'Different', '0', '1', 65, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/122913.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/032084.jpg', 'ground_truth', '1', '0', 99, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087556_9.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087456_0.0_AE.png', 'Different', '1', '0', 62, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087393_9.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087393_0.0_AE.png', 'Same', '1', '0', 2, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087471_0.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087571_9.0_AE.png', 'Different', '0', '1', 75, 1],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087419_0.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_aggressive/087419_9.0_AE.png', 'Same', '0', '1', 27, 0]
        ];

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