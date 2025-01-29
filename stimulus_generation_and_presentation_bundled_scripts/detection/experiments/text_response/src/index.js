import { initJsPsych } from 'jspsych';

import htmlKeyboardResponse from '@jspsych/plugin-html-keyboard-response';
import surveyText from '@jspsych/plugin-survey-text';
import surveyHtmlForm from '@jspsych/plugin-survey-html-form';
import HtmlSliderResponsePlugin from '@jspsych/plugin-html-slider-response';
import fullscreen from '@jspsych/plugin-fullscreen';
import surveyMultiSelect from '@jspsych/plugin-survey-multi-select';
import surveyMultiChoice from '@jspsych/plugin-survey-multi-choice';
import videoKeyboardResponse from '@jspsych/plugin-video-keyboard-response';

import {
    PsyanimApp,
    PsyanimJsPsychPlugin,
    PsyanimJsPsychTrial,

    PsyanimFirebaseBrowserClient, // Firebase uncomment

    PsyanimJsPsychTrialLoader,
    PsyanimJsPsychTrialSelector,

    PsyanimJsPsychDataWriterExtension,

    PsyanimJsPsychExperimentPlayerSceneTemplate,
    PsyanimJsPsychExperimentLoadingSceneTemplate,

} from 'psyanim2';
///Users/f0053cz/Documents/psyanim_v2/subtlety/textv_July2024/5_final_task/src/index.js

import firebaseJsonConfig from '../firebase.config.json'; //Firebase uncomment
import text_list from './Intro/Intro_text.js';

import allTrials from './outputTrialCollection_168vids.json';

// import trialIDs from "./all_videos.json";
import trialIDs from "./all_chase_wander_vids_subtlety.json"
const metadata = trialIDs.varList;

import stimSets from './main/stimSets_textv.json'; // list of 60 combinations  of trials (in integers)
console.log('stimSets', stimSets)

console.log('metadata.length', metadata.length);
console.log('allTrials.length', allTrials.length);

stimSets = stimSets.file_ind;
console.log(stimSets.length) // should be 500

/**
 *  Handle user authentication and any other configuration
 */
const userID = 'subtlety_textv';
const experimentName = 'subtlety_chaseonly_textv';

/***  Setup Psyanim App*/
// PsyanimApp.Instance.config.registerScene(EmptyScene);
PsyanimApp.Instance.run();
PsyanimApp.Instance.setCanvasVisible(false);

/**  Setup PsyanimJsPsychPlugin */
PsyanimJsPsychPlugin.setUserID(userID);
PsyanimJsPsychPlugin.setExperimentName(experimentName);
PsyanimJsPsychPlugin.setClearConsoleOnNewTrial(false);

const firebaseClient = new PsyanimFirebaseBrowserClient(firebaseJsonConfig); //Firebase uncomment
PsyanimJsPsychPlugin.setDocumentWriter(firebaseClient);

var db = firebaseClient._initFirestore(); // Now you have the Firestore instance
// var db = firebase.firestore();
const docSnapShot = await db.collection('counters').doc('pageViews').get();
var oldCount = docSnapShot.data().count;

function incrementPageViewCounter() {
    var pageViewRef = db.collection('counters').doc('pageViews');
    return db.runTransaction((transaction) => {
            return transaction.get(pageViewRef).then((doc) => {
                if (!doc.exists) {
                    throw "Document does not exist!";
                }
                // Increment the count
                var newCount = doc.data().count + 1;
                transaction.update(pageViewRef, { count: newCount });
            });
        })
        .then(() => {
            console.log("Transaction successfully committed!");
        })
        .catch((error) => {
            console.log("Transaction failed: ", error);
        });
}

incrementPageViewCounter();

// const randomIndex = Math.floor(Math.random() * (stimSets.length - 1)); // pick a random row from the list (<500)
// console.log('randomIndex', randomIndex); // should be less than 500
// const stim_set = stimSets[randomIndex]; // a row of 7 trial indices
const stim_set = stimSets[oldCount]; // a row of 7 trial indices
// pop stimSets[randomIndex]

console.log('stim_set.length', stim_set.length); // should be 7
console.log(stim_set); // should be numbers between 0 and 139 (incl.)

const stim_set_IDs = stim_set.map(index => metadata[index]); //equivalent to metadata[stim_set]; i.e., get the 84 trials' trial IDs (trial IDs are the trial-metadata file names)
// IMPORTANT: metadata and stim_set should be drawn from files with stimuli in the same order
console.log('stim_set_IDs', stim_set_IDs)

const actual_stim = allTrials.filter(obj => stim_set_IDs.includes(obj.trialID)); // get the trial objects based on the trial IDs
console.log('actual_stim', actual_stim); // should be 70 trial objects - 7 chargeSpeed levels. Perfectly counterbalanced. !!!Yet to double-check this in the data!!!

const demo_stim = [{ // a wander stimulus that's not presented in the chase-only batch
    "trialID": "1c695094-312c-450a-ac7f-f994dabb958d",
    "sessionID": "31fbd383-7e86-476b-bf13-a4390267318e",
    "experimentName": "defaultExperimentName",
    "sceneKey": "wanderScene_pseudoPredator_spawnPoint550_colorgrey_rep6",
    "agentMetadata": [{
            "name": "pseudoPredator",
            "shapeParams": {
                "shapeType": "PSYANIM_SHAPE_CIRCLE",
                "color": 0,
                "radius": 12
            }
        },
        {
            "name": "pseudoPrey",
            "shapeParams": {
                "shapeType": "PSYANIM_SHAPE_CIRCLE",
                "color": 13421772,
                "radius": 12
            }
        }
    ],
    "excludeAgents": []
}]


// [{
//     "trialID": "7b253001-d3ac-4c1c-b6aa-fc60e3be73e4",
//     "sessionID": "894fa11f-bb12-4181-98b7-451776581fc6",
//     "experimentName": "video_check",
//     "sceneKey": "playfight_videocheck",
//     "agentMetadata": [{
//             "name": "pseudoPredator",
//             "shapeParams": {
//                 "shapeType": "PSYANIM_SHAPE_CIRCLE",
//                 "color": 0,
//                 "radius": 12
//             }
//         },
//         {
//             "name": "pseudoPrey",
//             "shapeParams": {
//                 "shapeType": "PSYANIM_SHAPE_CIRCLE",
//                 "color": 13421772,
//                 "radius": 12
//             }
//         }
//     ],
//     "excludeAgents": []
// }];

const allTrialIDs = demo_stim.concat(actual_stim);


// // setup logger for PsyanimDebug to write to firebase
// PsyanimDebug.logger = new PsyanimFirebaseLogger(PsyanimApp.Instance.sessionID, firebaseClient);

/**
 *  Setup jsPsych experiment
 */
const jsPsych = initJsPsych({
    show_progress_bar: true,
    //Firebase uncomment
    extensions: [{
        type: PsyanimJsPsychDataWriterExtension,
        params: {
            documentWriter: firebaseClient,
            userID: userID,
            experimentName: experimentName
        }
    }],
    // on_finish: PsyanimJsPsychPlugin.handleExperimentFinished
});

// capture info from Prolific - for final v - don't delete!!!
var subject_id = jsPsych.data.getURLVariable('PROLIFIC_PID');
var study_id = jsPsych.data.getURLVariable('STUDY_ID');
var session_id = jsPsych.data.getURLVariable('SESSION_ID');

jsPsych.data.addProperties({
    subject_id: subject_id,
    study_id: study_id,
    session_id: session_id
});


let timeline = [];

// Prolific Id
timeline.push({
    type: surveyHtmlForm,
    preamble: '<p>Please enter your <strong>24-character</strong> Prolific ID here.</p>',
    html: '<p align="center"> <input id="prolific_ID_ID_start" class="textbox" type="text" size="50%" height="50" align="center" name="prolific_ID_name_start" required /> <br></p>',
    // on_load: function() {
    //     document.querySelector('input[type="text"]').className += " jspsych-input-text";
    // },
    on_load: function() {
        document.querySelector('input[type="text"]').style.cssText = 'font-size: 24px; width: 400px; height: 40px;';
    },
    on_finish: function() { console.log('testing on_finish') },
    extensions: [
        { type: PsyanimJsPsychDataWriterExtension }
    ]
});
//highlighting 24-character because people sometimes enter the experiment url :/

// 'Welcome' trial
timeline.push({
    type: htmlKeyboardResponse,
    stimulus: 'Welcome to the experiment.<br>Press <strong>Enter/Return</strong> to begin.',
    choices: ['enter'],
});


// Consent form 
timeline.push({
    type: surveyHtmlForm,
    preamble: text_list[0].consent_form_naturalistic,
    html: "<p><input type=checkbox id=consent_checkbox required/> <strong> I agree to take part in this study. </strong> </p>",
});

// Browser Preference 
timeline.push({
    type: surveyHtmlForm,
    preamble: text_list[1].devices_off_1,
    html: "  ",
    response_ends_trial: true,
    button_label: "Understood. Continue",
})

// turn off device 
timeline.push({
    type: surveyHtmlForm,
    preamble: text_list[1].devices_off_2,
    html: "  ",
    response_ends_trial: true,
    button_label: "All done. Continue",
})

//  Enter full screen 
timeline.push({
    type: fullscreen,
    fullscreen_mode: true,
    message: "<p>The experiment will run in full screen mode.<br>Press 'Continue' to proceed.</p>",
    data: {
        stimset_row: oldCount, // randomIndex,
        stim_set_IDs: stim_set_IDs,
        actual_stim: actual_stim
    },
    extensions: [
        { type: PsyanimJsPsychDataWriterExtension }
    ]
})


let experimentLoaderTrial = new PsyanimJsPsychTrial(PsyanimJsPsychExperimentLoadingSceneTemplate, PsyanimJsPsychExperimentLoadingSceneTemplate.key);
experimentLoaderTrial.setComponentParameter('experimentLoader',
    PsyanimJsPsychTrialLoader, 'trialInfo', allTrialIDs, false); // preload all trialIDs
experimentLoaderTrial.setComponentParameter('experimentLoader',
    PsyanimJsPsychTrialLoader, 'documentReader', firebaseClient, false);
timeline.push(experimentLoaderTrial.jsPsychTrialDefinition);


// Task Instruction
let sceneKey = "animationTestTrial";
let videoTestTrial = new PsyanimJsPsychTrial(PsyanimJsPsychExperimentPlayerSceneTemplate, sceneKey);
videoTestTrial.setComponentParameter('trialSelector', PsyanimJsPsychTrialSelector, 'trialInfo', demo_stim);
videoTestTrial.endTrialKeys = [];
videoTestTrial.endTrialOnPlaybackComplete = true;
timeline.push(videoTestTrial.jsPsychTrialDefinition);

// Define a function to check the time difference and possibly end the experiment
function checkTimeAndEndExperiment() {
    // Get data from the last two trials
    const lastTwoTrialsData = jsPsych.data.get().last(2).values();
    // Ensure we have two trials to compare
    if (lastTwoTrialsData.length === 2) {
        const previousTrial = lastTwoTrialsData[0];
        const currentTrial = lastTwoTrialsData[1];
        const timeDifference = currentTrial.time_elapsed - previousTrial.time_elapsed;
        if (timeDifference > 20000) { //change as needed
            PsyanimJsPsychPlugin.handleExperimentFinished;
            jsPsych.endExperiment("<p style='width:1000px;'>The video took over 20s to play out - with the current internet speed, you will not be able to complete this task.<br>Please message the experimenter providing details to be compensated for your time.</p>");

        }
    }
}

timeline.push({
    type: surveyHtmlForm,
    preamble: text_list[2].instructions_txt_a,
    html: ' ',
    response_ends_trial: true,
    button_label: "Continue",
    on_load: checkTimeAndEndExperiment
})

timeline.push({
    type: surveyHtmlForm,
    preamble: text_list[2].instructions_txt_b,
    html: ' ',
    response_ends_trial: true,
    button_label: "Continue",
})

// timeline.push({
//     type: surveyHtmlForm,
//     preamble: text_list[2].cover_story,
//     html: ' ',
//     response_ends_trial: true,
//     button_label: "Continue",
// })

// const response_text = '<p>1. <input type="text" id="Text1_id" name="Text1" size="40" required><br>2. <input type="text" id="Text2_id" name="Text2" size="40" required><br>3. <input type="text" id="Text3_id" name="Text3" size="40" required></p>';
const response_text = '<p><input type="text" id="respText_id" name="respText" size="40" required></p>';

// Describe the video you just saw using 1 word. <br>Guess if you do not know.</p>
timeline.push({
    type: surveyHtmlForm,
    preamble: text_list[3].instructions_q1,
    html: response_text,
    // autofocus: 'Text1_id',
    // on_load: function() {
    //     document.querySelector('input[type="text"]').className += " jspsych-input-text";
    // }
    on_load: function() {
        document.querySelector('input[type="text"]').style.cssText = 'font-size: 24px; width: 400px; height: 40px;';
    },
});

var response_page = {
    type: surveyHtmlForm,
    preamble: "<p style='width:1000px;'><strong>Briefly describe what the dots were doing.</strong><br>Guess if you do not know.",
    html: response_text,
    // autofocus: 'Text1_id'
    on_load: function() {
        document.querySelector('input[type="text"]').style.cssText = 'font-size: 24px; width: 400px; height: 40px;';
    },
}

// attention check
// var radioButtonTrial1 = {
//     type: surveyMultiChoice,
//     questions: [{
//         prompt: "<style>p{text-align: left;}</style><p><strong> Which of these do the dots represent?<br></strong></p>",
//         name: 'agents_type',
//         options: ['animals', 'balls', 'adults', 'children', 'magnets'],
//         required: false,
//         horizontal: false,
//     }],
//     on_finish: function(data) {
//         console.log('ON_FINISH. HERE!!')
//         var response = data.response.agents_type; //JSON.parse(data.responses).Q0;
//         console.log('response1', response)
//         if (response === 'children') {
//             // Correct answer
//             console.log('correct response on the 1st try')
//         } else {
//             // Incorrect answer
//             console.log('wrong response on the 1st try')
//         }
//     }
// }
// timeline.push(radioButtonTrial1)

// Function to check if the last response was correct
// function isLastResponseCorrect() {
//     var lastTrialData = jsPsych.data.get().last(1).values()[0];
//     console.log('lastTrialData', lastTrialData)
//         //return lastTrialData.correct === true;
//     return lastTrialData.response.agents_type === 'children';
// }

// var radioButtonTrial2 = {
//     type: surveyMultiChoice,
//     questions: [{
//         prompt: "<style>p{text-align: left;}</style>Miised/wrong response.<br><strong>One more chance to get this correct!!</strong><br><strong> Which of these did the dots represent?</strong></p>",
//         name: 'agents_type',
//         options: ['animals', 'balls', 'adults', 'children', 'magnets'],
//         required: false,
//         horizontal: false,
//     }],
//     on_finish: function(data) {
//         console.log('ON_FINISH. HERE!!')
//         var response = data.response.agents_type; //JSON.parse(data.responses).Q0;
//         console.log('response2', response)
//         if (response === 'children') {
//             // Correct answer
//             console.log('correct response on the 2nd try')
//                 //jsPsych.data.get().addToLastTrial({ correct: true });
//         } else {
//             // Incorrect answer
//             console.log('wrong response on the 2nd try')
//                 //jsPsych.data.get().addToLastTrial({ correct: false });
//         }
//     }
// }

// // Timeline conditional logic
// var conditionalNode1 = {
//     timeline: [radioButtonTrial2],
//     conditional_function: function() {
//         // Check if the last response was incorrect
//         return !isLastResponseCorrect(); //don't play if the last response was correct?
//     }
// };
// timeline.push(conditionalNode1);

// var quitting_message = {
//     type: fullscreen,
//     fullscreen_mode: false,
//     stimulus: "",
//     on_finish: function() {
//         console.log('quitting message')
//         jsPsych.endExperiment('Missed/wrong response the second time.<br>You are disqualified from taking part in this study.<br>Please message the experimenter to be compensated for your time.');
//     }
// };

// var conditionalNode2 = {
//     timeline: [quitting_message],
//     conditional_function: function() {
//         // Check if the last response was incorrect''
//         console.log('conditionalNode2')
//         return !isLastResponseCorrect(); //don't play if the last response was correct?
//     }
// };
// timeline.push(conditionalNode2);

timeline.push({
    type: htmlKeyboardResponse,
    stimulus: "<p style='width:1000px;'>Good job!<br> The main task will begin shortly. <br> Each scene is unrelated to the one before and after it, so try to not let your opinion about the dots in one animation influence the next one.<br>Press <strong>Enter/Return</strong> to proceed.</p>",
    choices: ['enter'],
    on_finish: function() { console.log('attention check finished.') }
})


/* ********************************** Main experiment starts here ************************************* */

// Main Scene trials 

for (let i = 0; i < 7; ++i) {

    let mainSceneKey = "mainTrial_" + i;
    let mainSceneTrial = new PsyanimJsPsychTrial(PsyanimJsPsychExperimentPlayerSceneTemplate, mainSceneKey);

    mainSceneTrial.setComponentParameter('trialSelector', PsyanimJsPsychTrialSelector, 'trialInfo', actual_stim);
    mainSceneTrial.endTrialKeys = [];
    mainSceneTrial.endTrialOnPlaybackComplete = true;
    timeline.push(mainSceneTrial.jsPsychTrialDefinition);
    timeline.push({
        type: htmlKeyboardResponse,
        stimulus: " ",
        trial_duration: 100,
        extensions: [
            { type: PsyanimJsPsychDataWriterExtension }
        ]
    })

    timeline.push(response_page);
    timeline.push({
        type: htmlKeyboardResponse,
        stimulus: " ",
        trial_duration: 100,
        extensions: [
            { type: PsyanimJsPsychDataWriterExtension }
        ]
    });
}

// // ********************************** Debrief #1 BEGIN HERE *************************************

/** Representation of the circles */
timeline.push({
    type: surveyMultiChoice,
    questions: [{
        prompt: "<style>p{text-align: left;}</style><p><strong>Which of the following did the dots represent?<br></strong></p>",
        name: 'agents_type',
        options: ['animals', 'balls', 'adults', 'children', 'magnets', 'something else (describe on the next page)'],
        required: true,
        horizontal: false
    }],
})

timeline.push({
    type: surveyHtmlForm,
    preamble: "<p align='left'> <strong> Press Enter/Return only when are done typing.</p>",
    html: '<p align="left">Did you use any strategies while performing this task?<br> <input id="strat" class="textbox" type="text" size="75%" height="60px" name="strategy" required /> <br></p><br><br><br>',
    on_load: function() {
        document.querySelector('input[type="text"]').style.cssText = 'font-size: 24px; width: 500px; height: 60px;';
    },
})

timeline.push({
    type: surveyHtmlForm,
    preamble: "<p align='left'> <strong> Press Enter/Return only when are done typing.</p>",
    html: '<p align="left">Any other feedback on the task?<br> <input id="fb" class="textbox" type="text" size="75%" height="60px" width="400px" name="feedback" required/></p><br><br><br>',
    on_load: function() {
        document.querySelector('input[type="text"]').style.cssText = 'font-size: 24px; width: 500px; height: 60px;';
    },
})

// ********************************** DEMOGRAPHIC BEGIN HERE *************************************
/** Transition to the demographic surveys */
// timeline.push({
//     type: surveyHtmlForm,
//     preamble: "Proceed to the next page for a few last questions.",
//     response_ends_trial: true,
//     html: ' ',
//     button_label: "Continue",
// })

/** Demographic question - Gender, hispanic */
timeline.push({
    type: surveyMultiChoice,
    questions: [{
        prompt: "<style>p{text-align: left;}</style><p><strong> How would you describe your gender? (optional)<br></strong></p>",
        name: 'gender',
        options: ['Male', 'Female', 'Non-binary/Other', 'Prefer not to say'],
        required: false,
        horizontal: true
    }, {
        prompt: "<style>p{text-align: left;}</style><p><strong> Are you of Hispanic or Latinx origin? (optional)<br></strong></p>",
        name: 'hispanic',
        options: ['Yes', 'No', 'Unknown/Prefer not to say'],
        required: false,
        horizontal: true
    }],
})

/** Demographic question - race */
timeline.push({
    type: surveyMultiSelect,
    questions: [{
        prompt: "<style>p{text-align: left;}</style><p><strong>How would you describe your race? (optional)<br></strong></p>",
        name: 'race',
        options: ['American Indian/Alaska Native', 'Asian', 'Native Hawaiian or Other Pacific Islander', 'Black or African American', 'White', 'More than one race', 'Unknown/Prefer not to say'],
        required: false,
        horizontal: false
    }]
})

/** Demographic question - age */
let demog_inp_qns = ['<p align="left">Age:<br> <input id = "age" type="number" style="width: 50%; height: 50px; font-size: 24px;" name="age" /> <br><br>Location (US state, optional): <br> <input id = "loc" type="text" style="width: 80%; height: 50px; font-size: 24px;" name="location"/></p><br><br><br>'];

timeline.push({
    type: surveyHtmlForm,
    preamble: '',
    html: demog_inp_qns[0],
    extensions: [
        { type: PsyanimJsPsychDataWriterExtension }
    ]
})


// ********************************** Debrief #2 BEGIN HERE *************************************
/** Feedback 2 */
// timeline.push({
//     type: surveyHtmlForm,
//     //preamble: "<p align='left'> <strong> Press Enter/Return only when are done typing.</p>",
//     html: '<p align="left">Is there anything you would like to tell us about this survey as a whole?<br> <input id = "fb2" class ="textbox" type="text" size="50%" height="50" name="feedback2" required/></p><br><br><br>',
//     extensions: [
//         { type: PsyanimJsPsychDataWriterExtension }
//     ]
// })

/** Exit full screen */
timeline.push({
    type: fullscreen,
    fullscreen_mode: false,
    message: "<p>Exitting fullscreen.<br>DO NOT CLOSE!! </p>"
})

// timeline.push({
//     type: surveyHtmlForm,
//     preamble: '<p>Please enter your <strong>24-character Prolific ID</strong> here again.</p>',
//     html: '<p align="center"> <input id = "prolific_ID_ID_end" class="textbox" type="text" size="50%"height="50" align="center" name="prolific_ID_name_end" required /> <br></p>',
//     extensions: [
//         { type: PsyanimJsPsychDataWriterExtension }
//     ]
// });


/** End trial */
timeline.push({
    type: htmlKeyboardResponse,
    stimulus: "<p style='text-align:center'>Congrats - you have completed the experiment!<br><strong> DO NOT CLOSE THIS WINDOW YET!</strong><br>Press any key to end this trial.</p>",
    on_load: PsyanimJsPsychPlugin.handleExperimentFinished,
    extensions: [
        { type: PsyanimJsPsychDataWriterExtension }
    ]
});


// REDIRECT BACK TO PROLIFIC
var goodbye = { // future studies - move window.location to on_finish (this still doesn't explain why data doesn't save though - because it did for some peopel
    type: htmlKeyboardResponse,
    stimulus: 'Going back to Prolific....',
    on_load: function() {
        window.location = 'https://app.prolific.com/submissions/complete?cc=CIDDFDBJ';
        //CHANGE THIS FOR NEW STUDIES!!!!
    }
};
timeline.push(goodbye);

jsPsych.run(timeline);