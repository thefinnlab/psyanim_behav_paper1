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

    PsyanimFirebaseBrowserClient,

    PsyanimJsPsychTrialLoader,
    PsyanimJsPsychTrialSelector,

    PsyanimJsPsychDataWriterExtension,

    PsyanimJsPsychExperimentPlayerSceneTemplate,
    PsyanimJsPsychExperimentLoadingSceneTemplate,

} from 'psyanim2';

import firebaseJsonConfig from '../firebase.config.json';

import EmptyScene from './EmptyScene.js';

// import userATrialIDs from './userATrialIDs.json' assert { type: 'json' };
// import userBTrialIDs from './userBTrialIDs.json' assert { type: 'json' };
// import userCTrialIDs from './userCTrialIDs.json' assert { type: 'json' };

import allTrials from './outputTrialCollection_168vids.json'; // trial collection objects of all 168 trial IDs
import metadata from "./all_videos_ed.js"; // all trial IDs
import stimSets from './stimSets.json'; // list of 500 combinations  of trials (in integers)
//console.log(stimSets)

console.log('metadata.length', metadata.length);
console.log('allTrials.length', allTrials.length);

stimSets = stimSets.file_ind;
console.log(stimSets.length) // should be 500
const randomIndex = Math.floor(Math.random() * (stimSets.length - 1)); // pick a random row from the list (<500)
console.log('randomIndex', randomIndex); // should be less than 500
const stim_set = stimSets[randomIndex]; // a row of 84 trial indices
console.log('stim_set.length', stim_set.length); // should be 84
console.log(stim_set); // should be numbers between 0 and 167 (incl.)

const stim_set_IDs = stim_set.map(index => metadata[index]); //equivalent to metadata[stim_set]; i.e., get the 84 trials' trial IDs (trial IDs are the trial-metadata file names)
//console.log('stim_set_IDs',stim_set_IDs)

const actual_stim = allTrials.filter(obj => stim_set_IDs.includes(obj.trialID)); // get the trial objects based on the trial IDs
console.log('actual_stim', actual_stim); // should be 84 trial objects - 36 chase, 36 mimic, 12 wander. Perfectly counterbalanced. !!!Yet to double-check this in the data!!!

const demo_stim = [{
        "trialID": "d4615934-e846-446e-813a-39c07c48e98d",
        "sessionID": "e484d903-8cda-4472-8704-3ca769540a7c",
        "experimentName": "defaultExperimentName",
        "sceneKey": "PredatorPrey_subtlety_30_pos250_col0",
        "agentMetadata": [{
                "name": "predator",
                "shapeParams": {
                    "shapeType": "PSYANIM_SHAPE_CIRCLE",
                    "color": 0,
                    "radius": 12
                }
            },
            {
                "name": "prey",
                "shapeParams": {
                    "shapeType": "PSYANIM_SHAPE_CIRCLE",
                    "color": 13421772,
                    "radius": 12
                }
            }
        ],
        "excludeAgents": []
    },
    {
        "trialID": "2fff4ff3-7628-4a4a-a82c-258b9549062c",
        "sessionID": "ec871a90-aafc-4789-8cd7-e037b30110aa",
        "experimentName": "defaultExperimentName",
        "sceneKey": "PredatorPreyMimic_mimic_150_preyPos474, 422_mimicPos250, 300",
        "agentMetadata": [{
                "name": "predator",
                "shapeParams": {
                    "shapeType": "PSYANIM_SHAPE_CIRCLE",
                    "color": 13421772,
                    "radius": 12
                }
            },
            {
                "name": "preyMimic",
                "shapeParams": {
                    "shapeType": "PSYANIM_SHAPE_CIRCLE",
                    "depth": 1,
                    "color": 0,
                    "radius": 12
                }
            },
            {
                "name": "prey",
                "shapeParams": {
                    "shapeType": "PSYANIM_SHAPE_CIRCLE",
                    "color": 16776960,
                    "radius": 12
                }
            }
        ],
        "excludeAgents": ["prey"]
    }
]

let allTrialIDs = demo_stim.concat(actual_stim);

// Randomly shuffle the array
const shuffle = (array) => {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
    console.log('shuffled traits:', array)
    return array;
};

const lbls_combinations = [
    ['Chasing', 'Moving independently'],
    ['Moving independently', 'Chasing']
]

let resp_randomNumber = Math.round(Math.random());
let lbls = lbls_combinations[resp_randomNumber];
console.log(resp_randomNumber, lbls); //0 ['Chasing', 'Moving independently'] or 1 ['Moving independently', 'Chasing']

/**
 *  Handle user authentication and any other configuration
 */
const userID = 'test_fullv';
const experimentName = 'full_expt_test';

/***  Setup Psyanim App*/
PsyanimApp.Instance.config.registerScene(EmptyScene);
PsyanimApp.Instance.run();
PsyanimApp.Instance.setCanvasVisible(false);

/**  Setup PsyanimJsPsychPlugin */
PsyanimJsPsychPlugin.setUserID(userID);
PsyanimJsPsychPlugin.setExperimentName(experimentName);
// PsyanimJsPsychPlugin.setClearConsoleOnNewTrial(false);

const firebaseClient = new PsyanimFirebaseBrowserClient(firebaseJsonConfig);
PsyanimJsPsychPlugin.setDocumentWriter(firebaseClient);

/**
 *  Setup jsPsych experiment
 */
const jsPsych = initJsPsych({
    show_progress_bar: true,
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

/** Prolific Id */
timeline.push({
    type: surveyHtmlForm,
    preamble: '<p><strong>Please enter your <strong>24-character</strong> Prolific ID here.</strong></p>',
    html: '<p align="center"> <input id="prolific_ID_ID_start" class="textbox" type="text" size="50%" height="50" align="center" name="prolific_ID_name_start" required /> <br></p>',
    //on_finish: function() { console.log('testing on_finish') }
    extensions: [
        { type: PsyanimJsPsychDataWriterExtension }
    ]
});
//highlighting 24-character because people sometimes enter the experiment url :/

/** 'Welcome' trial */
// timeline.push({
//     type: htmlKeyboardResponse,
//     stimulus: 'Welcome to the experiment.<br>Press Enter/Return to begin.',
//     choices: ['enter'],
// });

/** Consent form */
timeline.push({
    type: surveyHtmlForm,
    preamble: text_list[0].consent_form_naturalistic,
    html: "<p><input type=checkbox id=consent_checkbox required/> <strong> I agree to take part in this study. </strong> </p>",
});

/** Browser Preference */
timeline.push({
    type: surveyHtmlForm,
    preamble: text_list[1].devices_off_1,
    html: "  ",
    response_ends_trial: true,
    button_label: "Understood. Continue",
})

/** turn off device */
timeline.push({
    type: surveyHtmlForm,
    preamble: text_list[1].devices_off_2,
    html: "  ",
    response_ends_trial: true,
    button_label: "All done. Continue",
})

/** Enter full screen */
timeline.push({
    type: fullscreen,
    fullscreen_mode: true,
    message: "<p>The experiment will run in full screen mode.<br>Press 'Continue' to proceed.</p>",
    data: {
        stimset_row: randomIndex,
        response_sequence: lbls,
        stim_set_IDs: stim_set_IDs,
        actual_stim: actual_stim
    }
})

let experimentLoaderTrial = new PsyanimJsPsychTrial(PsyanimJsPsychExperimentLoadingSceneTemplate, PsyanimJsPsychExperimentLoadingSceneTemplate.key);
experimentLoaderTrial.setComponentParameter('experimentLoader',
    PsyanimJsPsychTrialLoader, 'trialInfo', allTrialIDs, false); // preload all trialIDs
experimentLoaderTrial.setComponentParameter('experimentLoader',
    PsyanimJsPsychTrialLoader, 'documentReader', firebaseClient, false);
timeline.push(experimentLoaderTrial.jsPsychTrialDefinition);


/** Task Instruction */
// timeline.push({
//     type: surveyHtmlForm,
//     preamble: text_list[2].instructions_txt_a,
//     html: ' ',
//     response_ends_trial: true,
//     button_label: "Continue",
// })

let sceneKey = "demo1_animationTestTrial";
let videoTestTrial = new PsyanimJsPsychTrial(PsyanimJsPsychExperimentPlayerSceneTemplate, sceneKey);
videoTestTrial.setComponentParameter('trialSelector', PsyanimJsPsychTrialSelector, 'trialInfo', demo_stim);
videoTestTrial.endTrialKeys = [];
videoTestTrial.endTrialOnPlaybackComplete = true;
timeline.push(videoTestTrial.jsPsychTrialDefinition);

timeline.push({
    type: surveyHtmlForm,
    preamble: "<strong>If you did NOT see the animation in the last page, exit now.<br>You will NOT be able to complete the survey!!</strong><br>If you did see it, press 'Continue'.<br> <br> ",
    html: ' ',
    response_ends_trial: true,
    button_label: "Continue",
})

/** General information */
timeline.push({
    type: surveyHtmlForm,
    preamble: text_list[1].general_intro,
    html: "  ",
    response_ends_trial: true,
    button_label: "Continue",
})

/** Cover Story */
timeline.push({
        type: surveyHtmlForm,
        preamble: text_list[2].cover_story,
        html: ' ',
        response_ends_trial: true,
        button_label: "Continue",
    })
    /** Task Instruction 2 */
timeline.push({
    type: surveyHtmlForm,
    preamble: text_list[2].instructions_txt_b,
    html: '<p style="width:1000px;"><input type=checkbox id=instructions_checkbox required/> <strong> Understood </strong> </label></p>',
})

/** Practice Trial */
timeline.push({
    type: htmlKeyboardResponse,
    stimulus: '<p style="width: 1000px"> After each animation, you get two questions: <br><strong>(1) Was one of the circles chasing the other, or were they moving independently? </strong>You will indicate your choice on a continuous bar.<br/><br/> <strong>(2) Which dot was chasing the other?</strong><br>Just guess if you did not see a chase in this video. <br><br> We will move on to a practice trial now, press <strong>Enter/Return</strong> to start </p>',
    choices: 'enter',
})

/** Demos oldv: loading */
// timeline.push({
//     type: videoKeyboardResponse,
//     stimulus: [
//         '../src/Intro/demo_animations/demo_subt150.webm'
//     ],
//     trial_ends_after_video: true,
//     choices: "NO_KEYS",
//     on_start: function(trial) {
//         // Apply CSS styles to the video element
//         trial.css_classes = ['video-border']; // This class will be used for styling
//     }
// })

var response_page = {
    type: htmlKeyboardResponse,
    stimulus: `
        <div>
            <p><strong>Was one dot chasing the other, or were the two dots moving independently?</strong></p>
            <div class="slider">
                <input type="range" min="0" max="100" value="50" id="slider"><br>
                <label for="slider">` + lbls[0] + `</label>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
                <label for="slider">` + lbls[1] + `</label>
            </div>
            <p><br><br><strong>Which dot was chasing the other?</strong><br>Just guess if you did not see a chase in this video.</p>
            <div id="chasing_dot">
                <input type="radio" name="chasing_dot" id="black" value="black">
                <label for="black"><img id="black-img" src="../src/Intro/demo_img/black_circle.png" width="25" height="25"></label>
                &nbsp;&nbsp;&nbsp;&nbsp&nbsp;&nbsp;&nbsp;&nbsp;
                <input type="radio" name="chasing_dot" id="grey" value="grey">
                <label for="grey"><img id="grey-img" src="../src/Intro/demo_img/grey_circle.png" width="25" height="25"></label>
            </div>
        </div>
    `,
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


/** Demo Scene trials */
//demo_stim.length
for (let i = 0; i < 1; ++i) {
    let sceneKey = "demoTrial_" + i;
    let demoSceneTrial = new PsyanimJsPsychTrial(PsyanimJsPsychExperimentPlayerSceneTemplate, sceneKey);
    demoSceneTrial.setComponentParameter('trialSelector', PsyanimJsPsychTrialSelector, 'trialInfo', demo_stim);
    demoSceneTrial.endTrialKeys = [];
    demoSceneTrial.subtext = ''
    demoSceneTrial.endTrialOnPlaybackComplete = true;
    timeline.push(demoSceneTrial.jsPsychTrialDefinition);
    timeline.push(response_page);

    timeline.push({
        type: htmlKeyboardResponse,
        stimulus: " ",
        trial_duration: 100,
    });
}

timeline.push({
    type: htmlKeyboardResponse,
    stimulus: "<p>You have completed the practice round. <br>Press <strong>Enter/Return</strong> to start the actual experiment.</p>",
    choices: ['enter'],
    on_finish: function() { console.log('practice finished.') },
    extensions: [
        { type: PsyanimJsPsychDataWriterExtension }
    ]
})

var radioButtonTrial1 = {
    type: surveyMultiChoice,
    questions: [{
        prompt: "<style>p{text-align: left;}</style><p><strong> Which of these did the circles represent?<br></strong></p>",
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
        prompt: "<style>p{text-align: left;}</style>Miised/wrong response.<br><strong>One more chance to get this correct!!</strong><br><strong> Which of these did the circles represent?</strong></p>",
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
        jsPsych.endExperiment('Missed/wrong response the second time. You are disqualified from taking part in this study.<br>Please message the experimenter to be compensated for your time.');
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
    stimulus: "<p>Good job! <br>Press <strong>Enter/Return</strong> to start the actual experiment.</p>",
    choices: ['enter'],
    on_finish: function() { console.log('practice finished.') }
})


// ********************************** Main experiment starts here *************************************

// /** experiment loader scene trial */
/** Main Scene trials */
//for (let i = 0; i < predatorPreyTest.length; ++i) {
for (let i = 0; i < 84; ++i) {

    let sceneKey = "trial_" + i;
    let mainSceneTrial = new PsyanimJsPsychTrial(PsyanimJsPsychExperimentPlayerSceneTemplate, sceneKey);

    //mainSceneTrial.setComponentParameter('trialSelector', PsyanimJsPsychTrialSelector, 'trialIDs', predatorPreyTest);
    mainSceneTrial.setComponentParameter('trialSelector', PsyanimJsPsychTrialSelector, 'trialInfo', actual_stim);
    mainSceneTrial.endTrialKeys = [];
    mainSceneTrial.endTrialOnPlaybackComplete = true;
    timeline.push(mainSceneTrial.jsPsychTrialDefinition);
    timeline.push(response_page);
    timeline.push({
        type: htmlKeyboardResponse,
        stimulus: " ",
        trial_duration: 100,
        extensions: [
            { type: PsyanimJsPsychDataWriterExtension }
        ]
    });

    if ((i > 0) && (i % 7 == 0)) { // BREAK EVERY 7 TRIALS, ALSO SAVE DATA INTERIM
        var break_screen = {
            type: surveyHtmlForm,
            preamble: "Optional break. Press continue when you are ready to proceed.",
            response_ends_trial: true,
            html: ' ',
            button_label: "Continue",
        };
        timeline.push(break_screen); // PUSH BREAK SCREEN
    }
}

// // ********************************** Debrief #1 BEGIN HERE *************************************

/** Representation of the circles */
timeline.push({
    type: surveyMultiChoice,
    questions: [{
        prompt: "<style>p{text-align: left;}</style><p><strong> Which of the following did the circles represent?<br></strong></p>",
        name: 'agents_type',
        options: ['animals', 'balls', 'adults', 'children', 'magnets', 'something else (describe on the next page)'],
        required: false,
        horizontal: false
    }],
})

timeline.push({
    type: surveyHtmlForm,
    preamble: "<p align='left'> <strong> Press Enter/Return only when are done typing.</p>",
    html: '<p align="left">Did you use any strategies while performing this task?<br> <input id = "strat" class="textbox" type="text" size="50%" height="50" name="strategy" required /> <br></p><p align="left">Any other feedback on the task?<br> <input id = "fb" class ="textbox" type="text" size="50%" height="50" name="feedback" required/></p><br><br><br>'
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
            prompt: "<style>p{text-align: left;}</style><p style='width: 1000px'><strong>" + questionnaire[i] + "<br></strong></p>",
            name: page_tag + '_qn' + i,
            options: response_choices,
            required: false,
            horizontal: true
        })
    }
    return questions_list;
}


/** AQ: text_list_AQ*/

var AQ_page1 = {
    type: surveyMultiChoice,
    preamble: '<p style="width: 1000px;padding: 0 0 3em 0">New questionnaire begins...<br><br>' + text_list_AQ[0].instructions + '</p>',
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
    // on_load: function(data) {
    //     saveData_temp("_data", jsPsych.data.get().csv());
    //     save_temp_data_csv();
    // }
};

// /** SRS */

var srs_page1 = {
    type: surveyMultiChoice,
    preamble: '<p style="width: 1000px;padding: 0 0 3em 0">New questionnaire begins...<br><br>' + text_list_SRS[0].instructions + '</p>',
    questions: get_questions_list(text_list_SRS[1].questions, text_list_SRS[2].response_choices, 0, 13, '1_SRS'),
}
var srs_page2 = {
    type: surveyMultiChoice,
    preamble: '<p style="padding: 0 0 2em 0">...questionnaire continues<br><br></p>',
    questions: get_questions_list(text_list_SRS[1].questions, text_list_SRS[2].response_choices, 13, 26, '2_SRS'),
}
var srs_page3 = {
    type: surveyMultiChoice,
    preamble: '<p style="padding: 0 0 2em 0">...questionnaire continues<br><br></p>',
    questions: get_questions_list(text_list_SRS[1].questions, text_list_SRS[2].response_choices, 26, 39, '3_SRS'),
}
var srs_page4 = {
    type: surveyMultiChoice,
    preamble: '<p style="padding: 0 0 2em 0">...questionnaire continues<br><br></p>',
    questions: get_questions_list(text_list_SRS[1].questions, text_list_SRS[2].response_choices, 39, 52, '4_SRS'),
}
var srs_page5 = {
    type: surveyMultiChoice,
    preamble: '<p style="padding: 0 0 2em 0">...questionnaire continues.<br><br></p>',
    questions: get_questions_list(text_list_SRS[1].questions, text_list_SRS[2].response_choices, 52, 66, '5_SRS'),
}

// /** PANAS */

var panas_page1 = {
    type: surveyMultiChoice,
    preamble: '<p style="width: 1000px;padding: 0 0 3em 0">New questionnaire begins...<br><br>' + text_list_PANAS[0].instructions + '</p>',
    questions: get_questions_list(text_list_PANAS[1].questions, text_list_PANAS[2].response_choices, 0, 10, '1_PANAS'),
}
var panas_page2 = {
    type: surveyMultiChoice,
    preamble: '<p style="padding: 0 0 2em 0">...questionnaire continues<br><br></p>',
    questions: get_questions_list(text_list_PANAS[1].questions, text_list_PANAS[2].response_choices, 10, 21, '2_PANAS'),
}

// /** Primals */

var prim_page1 = {
    type: surveyMultiChoice,
    preamble: '<p style="width: 1000px;padding: 0 0 3em 0">New questionnaire begins...<br><br>' + text_list_primals[0].instructions + '</p>',
    // + '<br> TBD: ADD PLEASE TRY TO RESPOND TO ALL QUESTIONS?',
    questions: get_questions_list(text_list_primals[1].questions, text_list_primals[2].response_choices, 0, 6, '1_Primals'),
};

var prim_page2 = {
    type: surveyMultiChoice,
    preamble: '<p style="padding: 0 0 2em 0">...questionnaire continues<br><br></p>',
    questions: get_questions_list(text_list_primals[1].questions, text_list_primals[2].response_choices, 6, 12, '2_Primals')
};

var prim_page3 = {
    type: surveyMultiChoice,
    preamble: '<p style="padding: 0 0 2em 0">...questionnaire continues<br><br></p>',
    questions: get_questions_list(text_list_primals[1].questions, text_list_primals[2].response_choices, 12, 19, '3_Primals'),
};

/** append all survey pages */

let surveyPage = [
    [AQ_page1, AQ_page2, AQ_page3, AQ_page4, AQ_page5],
    [srs_page1, srs_page2, srs_page3, srs_page4, srs_page5],
    [panas_page1, panas_page2],
    [prim_page1, prim_page2, prim_page3]
]

timeline.push({
    type: htmlKeyboardResponse,
    stimulus: "<p> Congrats on finishing the first part of the survey! <br> You will now move on to the second part. <br> Press <strong> Enter/Return </strong>to continue.",
    choices: 'enter',
})

/** shuffling of the survery order */
let survey_shuffled = shuffle(surveyPage);
console.log(survey_shuffled);
for (let i = 0; i < survey_shuffled.length; ++i) {
    if (i > 0) { // display an intro page for later surveys
        timeline.push({
            type: surveyHtmlForm,
            preamble: "Moving on to the next questionnaire (#" + String(i + 1) + " of 4).<br>Keep it up!<p>Please take note of the new task instructions and the slight change in the response choices before you start answering questions.<br><br></p>",
            response_ends_trial: true,
            html: ' ',
            button_label: "Continue",
            extensions: [
                { type: PsyanimJsPsychDataWriterExtension }
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
        { type: PsyanimJsPsychDataWriterExtension }
    ]
})


// ********************************** Debrief #2 BEGIN HERE *************************************
/** Feedback 2 */
timeline.push({
    type: surveyHtmlForm,
    //preamble: "<p align='left'> <strong> Press Enter/Return only when are done typing.</p>",
    html: '<p align="left">Is there anything you would like to tell us about this survey as a whole?<br> <input id = "fb2" class ="textbox" type="text" size="50%" height="50" name="feedback2" required/></p><br><br><br>',
    extensions: [
        { type: PsyanimJsPsychDataWriterExtension }
    ]
})

/** Exit full screen */
timeline.push({
    type: fullscreen,
    fullscreen_mode: false,
    message: "<p>Exitting fullscreen.<br>DO NOT CLOSE!! </p>"
})

timeline.push({
    type: surveyHtmlForm,
    preamble: '<p>Please enter your <strong>24-character Prolific ID</strong> here again.</p>',
    html: '<p align="center"> <input id = "prolific_ID_ID_end" class="textbox" type="text" size="50%"height="50" align="center" name="prolific_ID_name_end" required /> <br></p>',
    extensions: [
        { type: PsyanimJsPsychDataWriterExtension }
    ]
});


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
        window.location = 'https://app.prolific.co/submissions/complete?cc=132A693C';
        //CHANGE THIS FOR NEW STUDIES!!!!
    }
};
timeline.push(goodbye);



jsPsych.run(timeline);
