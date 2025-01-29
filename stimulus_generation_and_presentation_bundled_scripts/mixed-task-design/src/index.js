import { initJsPsych } from 'jspsych';

import htmlKeyboardResponse from '@jspsych/plugin-html-keyboard-response';
// import surveyText from '@jspsych/plugin-survey-text';
import surveyHtmlForm from '@jspsych/plugin-survey-html-form';
import HtmlSliderResponsePlugin from '@jspsych/plugin-html-slider-response';
import fullscreen from '@jspsych/plugin-fullscreen';
import surveyMultiSelect from '@jspsych/plugin-survey-multi-select';
import surveyMultiChoice from '@jspsych/plugin-survey-multi-choice';
// import videoKeyboardResponse from '@jspsych/plugin-video-keyboard-response';

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


import firebaseJsonConfig from '../firebase.config.json'; //Firebase uncomment
import text_list from './Intro/Intro_text.js';
import text_list_AQ from "./surveys/AQ_all1.js";
import text_list_PANAS from "./surveys/PANAS_nonumbers.js";
import text_list_ucla from "./surveys/UCLA_Loneliness_nonumbers.js";
import text_list_neoffi from "./surveys/NEO-FFI_nonumbers.js";

import allTrials_subtlety from './outputTrialCollection_168vids.json';
import allTrials_playfight from './outputTrialCollection_140vids.json';
// import trialIDs from "./all_videos.js"; // all trial IDs 
// const metadata = trialIDs[0].varList;

import trialIDs_subtlety from "./expt_material/all_videos_ed_subtlety.js";
import trialIDs_playfight from "./expt_material/all_videos_playfight.json";
const metadata_subtlety = trialIDs_subtlety;
const metadata_playfight = trialIDs_playfight.varList;

import stimSets_subtlety from './expt_material/stimSets_subtlety.json'; // list of 50 or 500 combinations  of trials (in integers)
import stimSets_playfight from './expt_material/stimSets_playfight1.json'; // list of 50 or 500 combinations  of trials (in integers)

stimSets_subtlety = stimSets_subtlety.file_ind;
stimSets_playfight = stimSets_playfight.file_ind;

const lbls_combinations_1 = [
    ['Moving independently', 'Chasing'],
    ['Chasing', 'Moving independently']
];

const lbls_combinations_resp_prompt_1 = [
    // ['moving independently', 'chasing'],
    // ['chasing', 'moving independently']
    'Were the two dots <strong>moving independently</strong>, or was one dot <strong>chasing</strong> the other?',
    'Was one dot <strong>chasing</strong> the other, or were the two dots <strong>moving independently</strong>?'
];

const lbls_combinations_2 = [
    ['Fighting', 'Playing'],
    ['Playing', 'Fighting']
    // ['Negative<br>interaction', 'Positive<br>interaction'],
    // ['Positive<br>interaction', 'Negative<br>interaction']
];

const lbls_combinations_resp_prompt_2 = [
    ['fighting (negative)', 'playing (positive)'],
    ['playing (positive)', 'fighting (negative)']
];

const dots = [
    ['black', 'grey'],
    ['grey', 'black']
];

let resp_randomNumber = Math.round(Math.random());
let lbls_1 = lbls_combinations_1[resp_randomNumber];
let lbls_resp_prompt_1 = lbls_combinations_resp_prompt_1[resp_randomNumber];

resp_randomNumber = Math.round(Math.random());
let lbls_2 = lbls_combinations_2[resp_randomNumber];
let lbls_resp_prompt_2 = lbls_combinations_resp_prompt_2[resp_randomNumber];

resp_randomNumber = Math.round(Math.random());
let lbls_pred_color = dots[resp_randomNumber];


/**
 *  Handle user authentication and any other configuration
 */

let nTrialPerBlock = 14; //14;
// randomization - block sequences ( 131313 / 313131 )
if (Math.random() < 0.5) {
    var blockSeq = [1, 2, 1, 2, 1, 2]
} else {
    var blockSeq = [2, 1, 2, 1, 2, 1]
}
const userID = 'subtlety_playfight_traits';
const experimentName = 'full_expt_mixed';

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

const randomIndex_subt = Math.floor(Math.random() * (stimSets_subtlety.length - 1)); // pick a random row from the list (<500)
const stim_set_subtlety = stimSets_subtlety[randomIndex_subt]; // a row of 7 trial indices
const stim_set_IDs_subtlety = stim_set_subtlety.map(index => metadata_subtlety[index]); //equivalent to metadata[stim_set]; i.e., get the 84 trials' trial IDs (trial IDs are the trial-metadata file names)
const actual_stim_subtlety = allTrials_subtlety.filter(obj => stim_set_IDs_subtlety.includes(obj.trialID)); // get the trial objects based on the trial IDs

const randomIndex_pf = Math.floor(Math.random() * (stimSets_playfight.length - 1)); // pick a random row from the list (<500)
const stim_set_playfight = stimSets_playfight[randomIndex_pf]; // a row of 7 trial indices
const stim_set_IDs_playfight = stim_set_playfight.map(index => metadata_playfight[index]); //equivalent to metadata[stim_set]; i.e., get the 84 trials' trial IDs (trial IDs are the trial-metadata file names)

const actual_stim_playfight = allTrials_playfight.filter(obj => stim_set_IDs_playfight.includes(obj.trialID)); // get the trial objects based on the trial IDs

import video_check from './video_check.json'
import demo_stim_subtlety from './demo_stim_subtlety.json'
import demo_stim_playfight from './demo_stim_playfight.json'

const allTrialIDs = video_check.concat(demo_stim_subtlety, demo_stim_playfight, actual_stim_subtlety, actual_stim_playfight);

// Randomly shuffle the array
const shuffle = (array) => {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
    console.log('shuffled traits:', array)
    return array;
};
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
            experimentName: experimentName,
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
});

// turn off device 
timeline.push({
    type: surveyHtmlForm,
    preamble: text_list[1].devices_off_2,
    html: "  ",
    response_ends_trial: true,
    button_label: "All done. Continue",
});

// var startTime, endTime;

//  Enter full screen 
timeline.push({
    type: fullscreen,
    fullscreen_mode: true,
    message: "<p>The experiment will run in full screen mode.<br>Press <b>Continue</b> to proceed.</p>",
    data: {
        stimset_row_subt: randomIndex_subt, // oldCount
        stimset_row_pf: randomIndex_pf, // oldCount
        blockSeq: blockSeq,
        stim_set_IDs_subtlety: stim_set_IDs_subtlety,
        stim_set_IDs_playfight: stim_set_IDs_playfight,
        actual_stim_subtlety: actual_stim_subtlety,
        actual_stim_playfight: actual_stim_playfight,
        lbls_1_subtlety: lbls_1,
        lbls_2_playfight: lbls_2,
        lbls_pred_color: lbls_pred_color
    },
    extensions: [
        { type: PsyanimJsPsychDataWriterExtension }
    ],
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
videoTestTrial.setComponentParameter('trialSelector', PsyanimJsPsychTrialSelector, 'trialInfo', video_check);
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
});

/** General information */
timeline.push({
    type: surveyHtmlForm,
    preamble: text_list[1].general_intro,
    html: "  ",
    response_ends_trial: true,
    button_label: "Continue",
})


timeline.push({
    type: surveyHtmlForm,
    preamble: text_list[2].cover_story,
    html: ' ',
    response_ends_trial: true,
    button_label: "Continue",
})

timeline.push({
    type: surveyHtmlForm,
    preamble: text_list[2].instructions_txt_b,
    html: ' ',
    response_ends_trial: true,
    button_label: "Continue",
})

timeline.push({
    type: surveyHtmlForm,
    preamble: text_list[2].task_intro,
    html: ' ',
    response_ends_trial: true,
    button_label: "Continue",
})

const slider_page_width = '1200'
var response_page_1 = {
    type: htmlKeyboardResponse,
    stimulus: `
        <div style='width:` + slider_page_width + `px;'>
            <p>` + lbls_resp_prompt_1 + `</p>
            <div class="slider">
                <input type="range" min="0" max="100" value="50" id="slider"><br>
                <label for="slider">` + lbls_1[0] + `</label>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
                <label for="slider">` + lbls_1[1] + `</label>
            </div>
            <p><br><br>Which dot was chasing the other?<br>Guess if you did not see a chase here.</p>
            <div id="chasing_dot">
                <input type="radio" name="chasing_dot" id=` + lbls_pred_color[0] + ` value="` + lbls_pred_color[0] + `">
                <label for="` + lbls_pred_color[0] + `"><img id="` + lbls_pred_color[0] + `-img" src="../src/Intro/demo_img/` + lbls_pred_color[0] + `_circle.png" width="25" height="25"></label>
                &nbsp;&nbsp;&nbsp;&nbsp&nbsp;&nbsp;&nbsp;&nbsp;
                <input type="radio" name="chasing_dot" id="` + lbls_pred_color[1] + `" value="` + lbls_pred_color[1] + `">
                <label for="` + lbls_pred_color[1] + `"><img id="` + lbls_pred_color[1] + `-img" src="../src/Intro/demo_img/` + lbls_pred_color[1] + `_circle.png" width="25" height="25"></label>
            </div>
        </div>
    `,

    /*            
        <div id="chasing_dot">
        <input type="radio" name="chasing_dot" id="black" value="black">
        <label for="black"><img id="black-img" src="../src/Intro/demo_img/black_circle.png" width="25" height="25"></label>
        &nbsp;&nbsp;&nbsp;&nbsp&nbsp;&nbsp;&nbsp;&nbsp;
        <input type="radio" name="chasing_dot" id="grey" value="grey">
        <label for="grey"><img id="grey-img" src="../src/Intro/demo_img/grey_circle.png" width="25" height="25"></label>
        </div>
    */

    choices: 'NO_KEYS',
    trial_duration: 10000,
    on_load: function() {
        const sliderElement = document.getElementById('slider');
        const blackRadio = document.getElementById('black');
        const greyRadio = document.getElementById('grey');

        // Event listener for slider
        sliderElement.addEventListener('input', function() {
            console.log('Slider Response:', sliderElement.value);
        });

        // Event listener for radio buttons
        blackRadio.addEventListener('change', function() {
            console.log('Button:', blackRadio.value, 'is clicked');
        });

        greyRadio.addEventListener('change', function() {
            console.log('Button:', greyRadio.value, 'is clicked');
        });

        // Finish trial when both questions are answered
        const finishTrialIfAnswered = function() {
            if (sliderElement.value !== '50' && (blackRadio.checked || greyRadio.checked)) {
                console.log('end with response');
                if (blackRadio.checked) {
                    let data_object = {
                        slider_response: sliderElement.value,
                        button_response: blackRadio.value,
                    };
                    jsPsych.data.write(data_object)
                } else {
                    let data_object = {
                        slider_response: sliderElement.value,
                        button_response: greyRadio.value,
                    };
                    jsPsych.data.write(data_object)
                }
                jsPsych.finishTrial();
            } else {
                // alert('Please answer both questions.');
            }
        };
        // Attach event listeners to check for both questions answered
        sliderElement.addEventListener('input', finishTrialIfAnswered);
        blackRadio.addEventListener('change', finishTrialIfAnswered);
        greyRadio.addEventListener('change', finishTrialIfAnswered);
    },
};

// const response_page_text = "<p style='width:700px;'>Did it seem like the dots were <br>playing (positive) or fighting (negative)?<br><br>";
const response_page_text_2 = "<p style='width:" + slider_page_width + "px;'>Did it seem like the dots were <br><strong>" + lbls_resp_prompt_2[0] + "</strong> or <strong>" + lbls_resp_prompt_2[1] + "</strong>?<br><br>";

var response_page_demo_2 = {
    type: HtmlSliderResponsePlugin,
    stimulus: response_page_text_2,
    labels: lbls_2,
    trial_duration: 10000,
    slider_width: 700,
    require_movement: true
}

// style='width:1000px;'

var response_page_2 = {
    type: HtmlSliderResponsePlugin,
    stimulus: response_page_text_2,
    labels: lbls_2,
    trial_duration: 10000,
    slider_width: 700,
    require_movement: true
}

timeline.push({
    type: surveyHtmlForm,
    preamble: text_list[2].demo_intro_subtlety,
    html: ' ',
    response_ends_trial: true,
    button_label: "Continue",
})

// Subtlety demo
for (let i = 0; i < 2; ++i) {
    let demoSceneKey = "demoTrial_subtlety";
    let demoSceneTrial = new PsyanimJsPsychTrial(PsyanimJsPsychExperimentPlayerSceneTemplate, demoSceneKey);
    demoSceneTrial.setComponentParameter('trialSelector', PsyanimJsPsychTrialSelector, 'trialInfo', demo_stim_subtlety);
    demoSceneTrial.endTrialKeys = [];
    demoSceneTrial.subtext = ''
    demoSceneTrial.endTrialOnPlaybackComplete = true;
    timeline.push(demoSceneTrial.jsPsychTrialDefinition);
    timeline.push({
        type: htmlKeyboardResponse,
        stimulus: " ",
        trial_duration: 100,
        on_load: checkTimeAndEndExperiment
    });

    timeline.push(response_page_1);

    timeline.push({
        type: htmlKeyboardResponse,
        stimulus: " ",
        trial_duration: 100,
    });
}

timeline.push({
    type: surveyHtmlForm,
    preamble: text_list[2].demo_intro_playfight,
    html: ' ',
    response_ends_trial: true,
    button_label: "Continue",
})

// playfight demo
for (let i = 0; i < 2; ++i) {
    let demoSceneKey = "demoTrial_playfight";
    let demoSceneTrial = new PsyanimJsPsychTrial(PsyanimJsPsychExperimentPlayerSceneTemplate, demoSceneKey);
    demoSceneTrial.setComponentParameter('trialSelector', PsyanimJsPsychTrialSelector, 'trialInfo', demo_stim_playfight);
    demoSceneTrial.endTrialKeys = [];
    demoSceneTrial.subtext = ''
    demoSceneTrial.endTrialOnPlaybackComplete = true;
    timeline.push(demoSceneTrial.jsPsychTrialDefinition);
    timeline.push({
        type: htmlKeyboardResponse,
        stimulus: " ",
        trial_duration: 100,
        on_load: checkTimeAndEndExperiment
    });

    timeline.push(response_page_demo_2);

    timeline.push({
        type: htmlKeyboardResponse,
        stimulus: " ",
        trial_duration: 100,
    });
}

timeline.push({
    type: htmlKeyboardResponse,
    stimulus: "<p style='width:1000px;'>You have completed both demos! <br>Press <strong>Enter/Return</strong> to proceed.</p>",
    choices: ['enter'],
    on_finish: function() { console.log('practice finished.') },
    extensions: [
        { type: PsyanimJsPsychDataWriterExtension }
    ]
})

timeline.push({
    type: htmlKeyboardResponse,
    stimulus: "<p style='width:1000px;'>IMPORTANT<br>If you see a static screen for a long time or if the videos are too slow (i.e., much much longer than 8s) in the next 20min, please return the survey and let us know for a small compensation.<br><br>Press <strong>Enter/Return</strong> to proceed.</p>",
    choices: ['enter'],
    on_finish: function() { console.log('practice finished.') },
    extensions: [
        // { type: PsyanimJsPsychDataWriterExtension }
    ]
})

// attention check
var radioButtonTrial1 = {
    type: surveyMultiChoice,
    questions: [{
        prompt: "<style>p{text-align: left;}</style><p><strong> Which of these did the dots represent?<br></strong></p>",
        name: 'agents_type',
        options: ['animals', 'balls', 'adults', 'children', 'magnets'],
        required: false,
        horizontal: false,
    }],
    on_finish: function(data) {
        console.log('ON_FINISH. HERE!!')
        var response = data.response.agents_type; //JSON.parse(data.responses).Q0;
        console.log('response1', response)
        if (response === 'children') {
            // Correct answer
            console.log('correct response on the 1st try')
        } else {
            // Incorrect answer
            console.log('wrong response on the 1st try')
        }
    }
}
timeline.push(radioButtonTrial1)

// Function to check if the last response was correct
function isLastResponseCorrect() {
    var lastTrialData = jsPsych.data.get().last(1).values()[0];
    console.log('lastTrialData', lastTrialData)
        //return lastTrialData.correct === true;
    return lastTrialData.response.agents_type === 'children';
}

var radioButtonTrial2 = {
    type: surveyMultiChoice,
    questions: [{
        prompt: "<style>p{text-align: left;}</style>Missed/wrong response.<br><strong>One more chance to get this correct!!</strong><br><strong> Which of these did the dots represent?</strong></p>",
        name: 'agents_type',
        options: ['animals', 'balls', 'adults', 'children', 'magnets'],
        required: false,
        horizontal: false,
    }],
    on_finish: function(data) {
        console.log('ON_FINISH. HERE!!')
        var response = data.response.agents_type; //JSON.parse(data.responses).Q0;
        console.log('response2', response)
        if (response === 'children') {
            // Correct answer
            console.log('correct response on the 2nd try')
                //jsPsych.data.get().addToLastTrial({ correct: true });
        } else {
            // Incorrect answer
            console.log('wrong response on the 2nd try')
                //jsPsych.data.get().addToLastTrial({ correct: false });
        }
    }
}

// Timeline conditional logic
var conditionalNode1 = {
    timeline: [radioButtonTrial2],
    conditional_function: function() {
        // Check if the last response was incorrect
        return !isLastResponseCorrect(); //don't play if the last response was correct?
    }
};
timeline.push(conditionalNode1);

var quitting_message = {
    type: fullscreen,
    fullscreen_mode: false,
    stimulus: "",
    on_finish: function() {
        console.log('quitting message')
        PsyanimJsPsychPlugin.handleExperimentFinished;
        jsPsych.endExperiment('Missed/wrong response the second time.<br>You are disqualified from taking part in this study.<br>Please message the experimenter to be compensated for your time.');
    }
};

var conditionalNode2 = {
    timeline: [quitting_message],
    conditional_function: function() {
        // Check if the last response was incorrect''
        console.log('conditionalNode2')
        return !isLastResponseCorrect(); //don't play if the last response was correct?
    }
};
timeline.push(conditionalNode2);


timeline.push({
    type: htmlKeyboardResponse,
    stimulus: "<p style='width:1000px;'>Good job!!<br><br>The main task will begin shortly.<br><br>Each scene is unrelated to the one before and after it. <br><br>Try to not let your opinion about the dots in one video influence the next.<br><br>Press <strong>Enter/Return</strong> to proceed.</p>",
    choices: ['enter'],
    on_finish: function() { console.log('attention check finished.') }
})


/* ********************************** Main experiment starts here ************************************* */

// Main Scene trials 

// for (let i = 0; i < 1; ++i) {

for (let iblock = 0; iblock < blockSeq.length; ++iblock) {

    if (blockSeq[iblock] == 1) {
        // subtlety block instructions
        timeline.push({
            type: surveyHtmlForm,
            preamble: "<p><u>Section " + String(iblock + 1) + " (situation 1)</u>" + text_list[3].main_task_intro_subtlety,
            html: "  ",
            response_ends_trial: true,
            button_label: "Start",
        })

        for (let i = 0; i < nTrialPerBlock; ++i) {

            let mainSceneKey = "subtlety_block" + iblock + "_trial_" + i;
            let mainSceneTrial = new PsyanimJsPsychTrial(PsyanimJsPsychExperimentPlayerSceneTemplate, mainSceneKey);

            mainSceneTrial.setComponentParameter('trialSelector', PsyanimJsPsychTrialSelector, 'trialInfo', actual_stim_subtlety);
            mainSceneTrial.endTrialKeys = [];
            mainSceneTrial.endTrialOnPlaybackComplete = true;
            timeline.push(mainSceneTrial.jsPsychTrialDefinition);
            timeline.push({
                type: htmlKeyboardResponse,
                stimulus: " ",
                trial_duration: 100,
                extensions: [
                    // { type: PsyanimJsPsychDataWriterExtension }
                ]
            })

            timeline.push(response_page_1);
            timeline.push({
                type: htmlKeyboardResponse,
                stimulus: " ",
                trial_duration: 100,
                extensions: [
                    { type: PsyanimJsPsychDataWriterExtension }
                ]
            });
        }
    }

    if (blockSeq[iblock] == 2) {
        // playfight block instructions
        timeline.push({
            type: surveyHtmlForm,
            preamble: "<p><u>Section " + String(iblock + 1) + " (situation 2) </u>" + text_list[3].main_task_intro_playfight,
            html: "  ",
            response_ends_trial: true,
            button_label: "Start",
        })

        for (let i = 0; i < nTrialPerBlock; ++i) {

            let mainSceneKey = "playfight_block" + iblock + "_trial_" + i;
            let mainSceneTrial = new PsyanimJsPsychTrial(PsyanimJsPsychExperimentPlayerSceneTemplate, mainSceneKey);

            mainSceneTrial.setComponentParameter('trialSelector', PsyanimJsPsychTrialSelector, 'trialInfo', actual_stim_playfight);
            mainSceneTrial.endTrialKeys = [];
            mainSceneTrial.endTrialOnPlaybackComplete = true;
            timeline.push(mainSceneTrial.jsPsychTrialDefinition);
            timeline.push({
                type: htmlKeyboardResponse,
                stimulus: " ",
                trial_duration: 100,
                extensions: [
                    // { type: PsyanimJsPsychDataWriterExtension }
                ]
            })

            timeline.push(response_page_2);
            timeline.push({
                type: htmlKeyboardResponse,
                stimulus: " ",
                trial_duration: 100,
                extensions: [
                    { type: PsyanimJsPsychDataWriterExtension }
                ]
            });
        }
    }

    // break screen after each block
    var break_screen = {
        type: surveyHtmlForm,
        preamble: "Optional break.<br> Press <b>Continue</b> when you are ready to proceed.<br>",
        response_ends_trial: true,
        html: ' ',
        button_label: "Continue",
        extensions: [
            { type: PsyanimJsPsychDataWriterExtension }
        ]
    };
    timeline.push(break_screen); // PUSH BREAK SCREEN


}

// // ********************************** Debrief #1 BEGIN HERE *************************************

/** Representation of the circles */
timeline.push({
    type: surveyMultiChoice,
    questions: [{
        prompt: "<style>p{text-align: left;}</style><p><strong> Which of the following did the dots represent?<br></strong></p>",
        name: 'agents_type',
        options: ['animals', 'balls', 'adults', 'children', 'magnets', 'something else (describe on the next page)'],
        required: false,
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

// ********************************** SURVEYS BEGIN HERE *************************************
/** Function that loads questions */
function get_questions_list(questionnaire, response_choices, qmin, qmax, page_tag) {
    // each page of the survey
    let questions_list = []
    console.log(qmin, qmax)
    let i = qmin;
    for (i = qmin; i < qmax; i++) {
        questions_list.push({
            prompt: "<style>p{text-align: left;}</style><p style='width:1500px'><strong>" + questionnaire[i] + "<br></strong></p>",
            name: page_tag + '_qn' + i,
            options: response_choices,
            required: false,
            horizontal: true
        })
    }
    return questions_list;
}

/** AQ */
var AQ_page1 = {
    type: surveyMultiChoice,
    preamble: '<p style="width:1500px;padding: 0 0 3em 0">New questionnaire begins...<br><br>' + text_list_AQ[0].instructions + '</p>',
    questions: get_questions_list(text_list_AQ[1].questions, text_list_AQ[2].response_choices, 0, 10, '1_AQ'),
};

var AQ_page2 = {
    type: surveyMultiChoice,
    preamble: '<p style="padding: 0 0 2em 0">...questionnaire continues<br><br></p>',
    questions: get_questions_list(text_list_AQ[1].questions, text_list_AQ[2].response_choices, 10, 20, '2_AQ'),
};

var AQ_page3 = {
    type: surveyMultiChoice,
    preamble: '<p style="padding: 0 0 2em 0">...questionnaire continues<br><br></p>',
    questions: get_questions_list(text_list_AQ[1].questions, text_list_AQ[2].response_choices, 20, 30, '3_AQ'),
};

var AQ_page4 = {
    type: surveyMultiChoice,
    preamble: '<p style="padding: 0 0 2em 0">...questionnaire continues<br><br></p>',
    questions: get_questions_list(text_list_AQ[1].questions, text_list_AQ[2].response_choices, 30, 40, '4_AQ'),
};

var AQ_page5 = {
    type: surveyMultiChoice,
    preamble: '<p style="padding: 0 0 2em 0">...questionnaire continues<br><br></p>',
    questions: get_questions_list(text_list_AQ[1].questions, text_list_AQ[2].response_choices, 40, 51, '5_AQ'),
};

// /** Panas */
var panas_page1 = {
    type: surveyMultiChoice,
    preamble: '<p style="width:1500px;padding: 0 0 3em 0">New questionnaire begins...<br><br>' + text_list_PANAS[0].instructions + '</p>',
    questions: get_questions_list(text_list_PANAS[1].questions, text_list_PANAS[2].response_choices, 0, 10, '1_PANAS'),
}
var panas_page2 = {
    type: surveyMultiChoice,
    preamble: '<p style="padding: 0 0 2em 0">...questionnaire continues<br><br></p>',
    questions: get_questions_list(text_list_PANAS[1].questions, text_list_PANAS[2].response_choices, 10, 21, '2_PANAS'),
}

/** UCLA_loneliness */
var UCLA_page1 = {
    type: surveyMultiChoice,
    preamble: '<p style="width:1500px;padding: 0 0 3em 0">New questionnaire begins...<br><br>' + text_list_ucla[0].instructions + '</p>',
    questions: get_questions_list(text_list_ucla[1].questions, text_list_ucla[2].response_choices, 0, 10, '1_UCLA'),
};

var UCLA_page2 = {
    type: surveyMultiChoice,
    preamble: '<p style="padding: 0 0 2em 0">...questionnaire continues<br><br></p>',
    questions: get_questions_list(text_list_ucla[1].questions, text_list_ucla[2].response_choices, 10, 21, '2_UCLA'),
};

// /** NEO-FFI */
var ffi_page1 = {
    type: surveyMultiChoice,
    preamble: '<p style="width:1500px;padding: 0 0 3em 0">New questionnaire begins...<br><br>' + text_list_neoffi[0].instructions,
    questions: get_questions_list(text_list_neoffi[1].questions, text_list_neoffi[2].response_choices, 0, 11, '1_ffi'),
}

var ffi_page2 = {
    type: surveyMultiChoice,
    preamble: '<p style="padding: 0 0 2em 0">...questionnaire continues<br><br></p>',
    questions: get_questions_list(text_list_neoffi[1].questions, text_list_neoffi[2].response_choices, 11, 21, '2_ffi'),
}

var ffi_page3 = {
    type: surveyMultiChoice,
    preamble: '<p style="padding: 0 0 2em 0">...questionnaire continues<br><br></p>',
    questions: get_questions_list(text_list_neoffi[1].questions, text_list_neoffi[2].response_choices, 21, 31, '3_ffi'),
}
var ffi_page4 = {
    type: surveyMultiChoice,
    preamble: '<p style="padding: 0 0 2em 0">...questionnaire continues<br><br></p>',
    questions: get_questions_list(text_list_neoffi[1].questions, text_list_neoffi[2].response_choices, 31, 41, '4_ffi'),
}
var ffi_page5 = {
    type: surveyMultiChoice,
    preamble: '<p style="padding: 0 0 2em 0">...questionnaire continues<br><br></p>',
    questions: get_questions_list(text_list_neoffi[1].questions, text_list_neoffi[2].response_choices, 41, 51, '5_ffi'),
}
var ffi_page6 = {
    type: surveyMultiChoice,
    preamble: '<p style="padding: 0 0 2em 0">...questionnaire continues<br><br></p>',
    questions: get_questions_list(text_list_neoffi[1].questions, text_list_neoffi[2].response_choices, 51, 61, '6_ffi'),
}

var fr_q1 = {
    type: surveyHtmlForm,
    preamble: '',
    html: '<p>Please estimate the number of <b>close friends</b> that you have,' +
        '<br>where "close friends" are people that you feel at ease with and can talk to about private matters.</p>' +
        '<p><input id = "friend" type="number"  min="-1" size="100%" height="50" name="friend"/></p>',
    on_load: function() {
        document.querySelector('input[type="number"]').style.cssText = 'font-size: 24px; width: 300px; height: 60px;';
    }
};


// /** append all survey pages */
let surveyPage = [
    [AQ_page1, AQ_page2, AQ_page3, AQ_page4, AQ_page5],
    [panas_page1, panas_page2],
    [UCLA_page1, UCLA_page2],
    [ffi_page1, ffi_page2, ffi_page3, ffi_page4, ffi_page5, ffi_page6],
    [fr_q1],
]

timeline.push({
    type: htmlKeyboardResponse,
    stimulus: "<p> Congrats on finishing the first part of the survey! <br> You will now move on to the second part. <br> Press <strong> Enter/Return </strong>to continue.",
    choices: 'enter',
})

// /** shuffling of the survery order */
let survey_shuffled = shuffle(surveyPage);
console.log(survey_shuffled);
for (let i = 0; i < survey_shuffled.length; ++i) {
    if (i > 0) { // display an intro page for later surveys
        timeline.push({
            type: surveyHtmlForm,
            preamble: '<p style="width:1000px;">Moving on to the next questionnaire (#' + String(i + 1) + ' of 5).<br>Keep it up!<br>Please take note of the new task instructions and the slight change in the response choices before you start answering questions.<br><br></p>',
            response_ends_trial: true,
            html: ' ',
            button_label: "Continue",
            extensions: [
                // { type: PsyanimJsPsychDataWriterExtension }
            ]
        })
    }
    for (let j = 0; j < survey_shuffled[i].length; ++j) {
        timeline.push(survey_shuffled[i][j])
    }
}

// ********************************** DEMOGRAPHIC BEGIN HERE *************************************
/** Transition to the demographic surveys */
timeline.push({
    type: surveyHtmlForm,
    preamble: "Proceed to the next page for a few last questions.",
    response_ends_trial: true,
    html: ' ',
    button_label: "Continue",
})

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
let demog_inp_qns = ['<p align="left">Age:<br> <input id = "age" type="number" size="50%" height="50" name="age" /> <br><br>Location (US state, optional): <br> <input id = "loc" type="text" size="50%" height="50" name="location"/></p><br><br><br>'];

timeline.push({
    type: surveyHtmlForm,
    preamble: '',
    html: demog_inp_qns[0],
    extensions: [
        { type: PsyanimJsPsychDataWriterExtension },
    ],
    // on_load: function() {
    //     document.querySelector('input[type="text"]').style.cssText = 'font-size: 24px; width: 500px; height: 60px;';
    // },
})


// ********************************** Debrief #2 BEGIN HERE *************************************
/** Feedback 2 */
timeline.push({
    type: surveyHtmlForm,
    //preamble: "<p align='left'> <strong> Press Enter/Return only when are done typing.</p>",
    html: '<p align="left">Is there anything you would like to tell us about this survey as a whole?<br> <input id = "fb2" class ="textbox" type="text" size="50%" height="50" name="feedback2" required/></p><br><br><br>',
    on_load: function() {
        document.querySelector('input[type="text"]').style.cssText = 'font-size: 24px; width: 500px; height: 60px;';
    },
    extensions: [
        // { type: PsyanimJsPsychDataWriterExtension }
    ]
})

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