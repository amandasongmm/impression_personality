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
			['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087481_2.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087581_8.0_AE.png', 'Different', '0', '1', 85, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087437_8.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087437_2.0_AE.png', 'Same', '1', '0', 44, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087543_8.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087439_2.0_AE.png', 'Different', '1', '0', 49, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087463_2.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087563_8.0_AE.png', 'Different', '0', '1', 68, 1],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087462_2.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087562_8.0_AE.png', 'Different', '0', '1', 67, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087428_8.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087428_2.0_AE.png', 'Same', '1', '0', 36, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087413_2.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087413_8.0_AE.png', 'Same', '0', '1', 21, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087566_8.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087466_2.0_AE.png', 'Different', '1', '0', 71, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087430_2.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087430_8.0_AE.png', 'Same', '0', '1', 38, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087585_8.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087485_2.0_AE.png', 'Different', '1', '0', 88, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087484_2.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087584_8.0_AE.png', 'Different', '0', '1', 87, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087486_2.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087534_8.0_AE.png', 'Different', '0', '1', 89, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/128593.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/038546.jpg', 'ground_truth', '1', '0', 92, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087478_2.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087578_8.0_AE.png', 'Different', '0', '1', 82, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087574_8.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087474_2.0_AE.png', 'Different', '1', '0', 78, 1],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087440_2.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087580_8.0_AE.png', 'Different', '0', '1', 46, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087392_8.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087392_2.0_AE.png', 'Same', '1', '0', 1, 1],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087448_2.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087557_8.0_AE.png', 'Different', '0', '1', 54, 1],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087551_8.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087451_2.0_AE.png', 'Different', '1', '0', 57, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087546_8.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087446_2.0_AE.png', 'Different', '1', '0', 52, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087405_2.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087405_8.0_AE.png', 'Same', '0', '1', 13, 1],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087399_2.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087399_8.0_AE.png', 'Same', '0', '1', 7, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087569_8.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087469_2.0_AE.png', 'Different', '1', '0', 73, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087415_8.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087415_2.0_AE.png', 'Same', '1', '0', 23, 1],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087558_8.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087458_2.0_AE.png', 'Different', '1', '0', 63, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087582_8.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087483_2.0_AE.png', 'Different', '1', '0', 86, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087417_2.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087417_8.0_AE.png', 'Same', '0', '1', 25, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087426_2.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087426_8.0_AE.png', 'Same', '0', '1', 34, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087577_8.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087477_2.0_AE.png', 'Different', '1', '0', 81, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087541_8.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087441_2.0_AE.png', 'Different', '1', '0', 47, 1],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087415_2.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087415_8.0_AE.png', 'Same', '0', '1', 23, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087424_2.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087424_8.0_AE.png', 'Same', '0', '1', 32, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087442_2.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087542_8.0_AE.png', 'Different', '0', '1', 48, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087550_8.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087450_2.0_AE.png', 'Different', '1', '0', 56, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087391_2.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087391_8.0_AE.png', 'Same', '0', '1', 0, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087436_2.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087436_8.0_AE.png', 'Same', '0', '1', 43, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087414_2.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087414_8.0_AE.png', 'Same', '0', '1', 22, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087465_2.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087468_8.0_AE.png', 'Different', '0', '1', 70, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087420_2.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087420_8.0_AE.png', 'Same', '0', '1', 28, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087401_2.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087401_8.0_AE.png', 'Same', '0', '1', 9, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087419_8.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087419_2.0_AE.png', 'Same', '1', '0', 27, 1],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087409_2.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087409_8.0_AE.png', 'Same', '0', '1', 17, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087418_8.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087418_2.0_AE.png', 'Same', '1', '0', 26, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087551_8.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087451_2.0_AE.png', 'Different', '1', '0', 57, 1],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087421_8.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087421_2.0_AE.png', 'Same', '1', '0', 29, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087553_8.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087453_2.0_AE.png', 'Different', '1', '0', 59, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087575_8.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087475_2.0_AE.png', 'Different', '1', '0', 79, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087471_2.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087571_8.0_AE.png', 'Different', '0', '1', 75, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087422_8.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087422_2.0_AE.png', 'Same', '1', '0', 30, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087438_2.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087538_8.0_AE.png', 'Different', '0', '1', 45, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087449_2.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087549_8.0_AE.png', 'Different', '0', '1', 55, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087574_8.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087474_2.0_AE.png', 'Different', '1', '0', 78, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087405_2.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087405_8.0_AE.png', 'Same', '0', '1', 13, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087416_2.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087416_8.0_AE.png', 'Same', '0', '1', 24, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087463_2.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087563_8.0_AE.png', 'Different', '0', '1', 68, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087397_8.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087397_2.0_AE.png', 'Same', '1', '0', 5, 1],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/163099.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/017814.jpg', 'ground_truth', '0', '1', 94, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087467_2.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087567_8.0_AE.png', 'Different', '0', '1', 72, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087397_2.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087397_8.0_AE.png', 'Same', '0', '1', 5, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087425_8.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087425_2.0_AE.png', 'Same', '1', '0', 33, 1],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087423_2.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087423_8.0_AE.png', 'Same', '0', '1', 31, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087433_2.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087433_8.0_AE.png', 'Same', '0', '1', 41, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087427_8.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087427_2.0_AE.png', 'Same', '1', '0', 35, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087434_8.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087434_2.0_AE.png', 'Same', '1', '0', 42, 1],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087434_8.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087434_2.0_AE.png', 'Same', '1', '0', 42, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087408_2.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087408_8.0_AE.png', 'Same', '0', '1', 16, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087425_8.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087425_2.0_AE.png', 'Same', '1', '0', 33, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087406_2.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087406_8.0_AE.png', 'Same', '0', '1', 14, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087540_8.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087480_2.0_AE.png', 'Different', '1', '0', 84, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087459_2.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087559_8.0_AE.png', 'Different', '0', '1', 64, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087413_2.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087413_8.0_AE.png', 'Same', '0', '1', 21, 1],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087431_2.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087431_8.0_AE.png', 'Same', '0', '1', 39, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087403_8.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087403_2.0_AE.png', 'Same', '1', '0', 11, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087454_2.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087554_8.0_AE.png', 'Different', '0', '1', 60, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/028757.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/063159.jpg', 'ground_truth', '0', '1', 93, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087447_2.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087547_8.0_AE.png', 'Different', '0', '1', 53, 1],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087412_8.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087412_2.0_AE.png', 'Same', '1', '0', 20, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087554_8.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087454_2.0_AE.png', 'Different', '1', '0', 60, 1],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087429_8.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087429_2.0_AE.png', 'Same', '1', '0', 37, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087476_2.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087568_8.0_AE.png', 'Different', '0', '1', 80, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/183037.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/105842.jpg', 'ground_truth', '0', '1', 98, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087544_8.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087444_2.0_AE.png', 'Different', '1', '0', 50, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/051409.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/109934.jpg', 'ground_truth', '1', '0', 90, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087534_8.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087486_2.0_AE.png', 'Different', '1', '0', 89, 1],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087441_2.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087541_8.0_AE.png', 'Different', '0', '1', 47, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087404_2.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087404_8.0_AE.png', 'Same', '0', '1', 12, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087445_2.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087545_8.0_AE.png', 'Different', '0', '1', 51, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087547_8.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087447_2.0_AE.png', 'Different', '1', '0', 53, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/150350.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/142336.jpg', 'ground_truth', '1', '0', 97, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087461_2.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087561_8.0_AE.png', 'Different', '0', '1', 66, 1],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087394_8.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087394_2.0_AE.png', 'Same', '1', '0', 3, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087472_2.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087572_8.0_AE.png', 'Different', '0', '1', 76, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087479_2.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087579_8.0_AE.png', 'Different', '0', '1', 83, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087473_2.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087573_8.0_AE.png', 'Different', '0', '1', 77, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087452_2.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087552_8.0_AE.png', 'Different', '0', '1', 58, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087450_2.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087550_8.0_AE.png', 'Different', '0', '1', 56, 1],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087396_8.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087396_2.0_AE.png', 'Same', '1', '0', 4, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087411_2.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087411_8.0_AE.png', 'Same', '0', '1', 19, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087410_2.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087410_8.0_AE.png', 'Same', '0', '1', 18, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087392_2.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087392_8.0_AE.png', 'Same', '0', '1', 1, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087570_8.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087470_2.0_AE.png', 'Different', '1', '0', 74, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087400_8.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087400_2.0_AE.png', 'Same', '1', '0', 8, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087542_8.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087442_2.0_AE.png', 'Different', '1', '0', 48, 1],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087402_8.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087402_2.0_AE.png', 'Same', '1', '0', 10, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087432_8.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087432_2.0_AE.png', 'Same', '1', '0', 40, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087407_2.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087407_8.0_AE.png', 'Same', '0', '1', 15, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/101536.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/042063.jpg', 'ground_truth', '0', '1', 96, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087455_2.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087555_8.0_AE.png', 'Different', '0', '1', 61, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087461_2.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087561_8.0_AE.png', 'Different', '0', '1', 66, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/065232.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/046077.jpg', 'ground_truth', '0', '1', 91, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087398_8.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087398_2.0_AE.png', 'Same', '1', '0', 6, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087557_8.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087448_2.0_AE.png', 'Different', '1', '0', 54, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087464_2.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087529_8.0_AE.png', 'Different', '0', '1', 69, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/059649.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/069891.jpg', 'ground_truth', '0', '1', 95, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087460_2.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087560_8.0_AE.png', 'Different', '0', '1', 65, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/058879.jpg', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/012584.jpg', 'ground_truth', '1', '0', 99, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087556_8.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087456_2.0_AE.png', 'Different', '1', '0', 62, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087393_8.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087393_2.0_AE.png', 'Same', '1', '0', 2, 0],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087471_2.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087571_8.0_AE.png', 'Different', '0', '1', 75, 1],
['https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087419_2.0_AE.png', 'https://raw.githubusercontent.com/amandasongmm/impression_personality/master/static/amt_trustworthy/087419_8.0_AE.png', 'Same', '0', '1', 27, 0]
				// ['https://github.com/amandasongmm/impression_personality/tree/master/static/img/087535_8.0_AE.png?raw=true', 'https://github.com/amandasongmm/impression_personality/tree/master/static/img/087435_2.0_AE.png?raw=true', 'test', 1, 0, 44, 0]
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