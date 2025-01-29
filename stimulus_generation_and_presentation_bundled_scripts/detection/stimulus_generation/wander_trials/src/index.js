import { initJsPsych } from 'jspsych';

import htmlKeyboardResponse from '@jspsych/plugin-html-keyboard-response';
import htmlSliderResponse from '@jspsych/plugin-html-slider-response';
// adapted from https://github.com/thefinnlab/psyanim-core-examples/blob/pseudoPredator-pseudoPrey-experiment/src/index.js
import {
    PsyanimApp,
    PsyanimJsPsychPlugin,
    PsyanimJsPsychTrial,
    PsyanimFirebaseBrowserClient,
} from 'psyanim2';

import firebaseJsonConfig from '../firebase.config.json';

import wanderScene from './wander.js';

/**
 *  Handle user authentication and any other configuration
 */
const userID = 'wander';
const experimentName = 'defaultExperimentName';

/**
 *  Setup Psyanim App
 */
PsyanimApp.Instance.config.registerScene(wanderScene);
PsyanimApp.Instance.run();

PsyanimApp.Instance.setCanvasVisible(false);

/**
 *  Setup PsyanimJsPsychPlugin
 */
PsyanimJsPsychPlugin.setUserID(userID);
PsyanimJsPsychPlugin.setExperimentName(experimentName);

const firebaseClient = new PsyanimFirebaseBrowserClient(firebaseJsonConfig);
PsyanimJsPsychPlugin.setDocumentWriter(firebaseClient);

/**
 *  Setup jsPsych experiment
 */
const jsPsych = initJsPsych({
    on_finish: PsyanimJsPsychPlugin.handleExperimentFinished
});

let timeline = [];

// 'Welcome' trial
timeline.push({
    type: htmlKeyboardResponse,
    stimulus: "<p style='text-align:center'>Welcome to the experiment.  Press any key to begin.</p>"
});

// test wander
//let wanderSceneTrial = new PsyanimJsPsychTrial(wanderScene, wanderScene.key);
//timeline.push(wanderSceneTrial.jsPsychTrialDefinition);

// In this world, we're going to randomly assign spawn locations to pseudoPredator and pseudoPrey for each trial

let nRep = 11; // we need 6 in the end (for 24 videos), generating 2x for that atm
let nTrials = nRep * 4; // 4 unique combinations.

let leftSpawnPoint = { x: 250, y: 300 };
let rightSpawnPoint = { x: 550, y: 300 };

let pseudoPredatorSpawnPoints = [];
let pseudoPreySpawnPoints = [];

let pseudoPredatorColors = [];
let pseudoPreyColors = [];

let colors_inds = ['black', 'grey']
let colors = { black: 0x000000, grey: 0xcccccc };


//0x000000: black; 0xcccccc: grey
for (let iter = 0; iter < nRep; ++iter) { // number of copies
    for (let i = 0; i < 2; ++i) { // i=0 -> pseudoPredator on the left, i = 1 -> pseudoPredator on the right.
        for (let j = 0; j < 2; ++j) { // j=0 -> pseudoPredator is black, i = 1 -> pseudoPredator is grey.
            let pseudoPredatorSpawnsOnLeft = i;
            let pseudoPredatorColorBlack = j;

            if (pseudoPredatorSpawnsOnLeft == 0 & pseudoPredatorColorBlack == 0) { // slower agent - black, left
                pseudoPredatorSpawnPoints.push(leftSpawnPoint);
                pseudoPredatorColors.push(colors_inds[0]); //(colors[colors_inds[0]]); //0x000000);
                pseudoPreySpawnPoints.push(rightSpawnPoint);
                pseudoPreyColors.push(colors_inds[1]); //((colors['grey']); //(0xcccccc); 

            } else if (pseudoPredatorSpawnsOnLeft == 0 & pseudoPredatorColorBlack == 1) { // slower agent - grey, left
                pseudoPredatorSpawnPoints.push(leftSpawnPoint);
                pseudoPredatorColors.push(colors_inds[1]); //((colors['grey']); //(0xcccccc);
                pseudoPreySpawnPoints.push(rightSpawnPoint);
                pseudoPreyColors.push(colors_inds[0]); //(colors['black']); //(0x000000);

            } else if (pseudoPredatorSpawnsOnLeft == 1 & pseudoPredatorColorBlack == 0) { // slower agent - black, right
                pseudoPredatorSpawnPoints.push(rightSpawnPoint);
                pseudoPredatorColors.push(colors_inds[0]); //(colors['black']); //(0x000000);
                pseudoPreySpawnPoints.push(leftSpawnPoint);
                pseudoPreyColors.push(colors_inds[1]); //((colors['grey']); //(0xcccccc);

            } else { // slower agent - grey, right
                pseudoPredatorSpawnPoints.push(rightSpawnPoint);
                pseudoPredatorColors.push(colors_inds[1]); //(colors['grey']); //(0xcccccc);

                pseudoPreySpawnPoints.push(leftSpawnPoint);
                pseudoPreyColors.push(colors_inds[0]); //(colors['black']); //(0x000000);
            }
        }
    }
}

//const speed = [.1, 10, .1, 10, .1, 10, .1, 10, .1, 10, .1, 10];
//const acceleration = [.1, 1, .1, 1, .1, 1, .1, 1, .1, 1, .1, 1];
//let nTrials_test = 4;
for (let i = 0; i < nTrials; ++i) {

    let rep = Math.floor(i / 4);
    let sceneKey = wanderScene.key + "_pseudoPredator_spawnPoint" + pseudoPredatorSpawnPoints[i]['x'] + "_color" + pseudoPreyColors[i] + "_rep" + rep;

    let trial = new PsyanimJsPsychTrial(wanderScene, sceneKey);

    // modify subtlety parameter for this scene
    trial.setEntityParameter('pseudoPredator', 'initialPosition', pseudoPredatorSpawnPoints[i]);
    trial.setEntityShapeParameter('pseudoPredator', 'color', colors[pseudoPredatorColors[i]]);

    trial.setEntityParameter('pseudoPrey', 'initialPosition', pseudoPreySpawnPoints[i]);
    trial.setEntityShapeParameter('pseudoPrey', 'color', colors[pseudoPreyColors[i]]);

    //trial.setPrefabParameter('pseudoPrey', 'maxSpeed', .5); //speed[i]);
    //trial.setPrefabParameter('pseudoPredator', 'maxSpeed', 10); // speed[i]);

    //trial.setPrefabParameter('pseudoPrey', 'maxAcceleration', .4); //acceleration[i]);
    //trial.setPrefabParameter('pseudoPredator', 'maxAcceleration', .1); //acceleration[i]);

    trial.duration = 6000;
    trial.endTrialKeys = [];

    //trial.recordAnimationParameter('pseudoPredator', PsyanimWanderBehavior,
    //    '_subtletyAngle', 'maxSpeed');

    trial.addAgentNamesToRecord(['pseudoPredator', 'pseudoPrey']);

    trial.subtext = 'trial' + i + ',<br>sceneKey=' + sceneKey + ',<br>slow agent start pos=' + String(pseudoPredatorSpawnPoints[i]['x']) + ',<br>slow agent color=' + pseudoPredatorColors[i]
        //+',<br>speed=' + speed[i]
        // +',<br>acceleration=' + acceleration[i]


    // can add custom data to be saved with the 'jsPsych.data'
    trial.jsPsychData = {
        sceneKey: sceneKey,
        pseudoPredatorInitialPosition: pseudoPredatorSpawnPoints[i],
        pseudoPreyInitialPosition: pseudoPreySpawnPoints[i]
    };

    timeline.push(trial.jsPsychTrialDefinition);

    // adding a SLIDER response after each trial
    /*timeline.push({
        type: htmlSliderResponse,
        stimulus: '<REPLACE W/ EXACT TEXT><br?Was one circle interacting the other, or were the circles moving independently?',
        require_movement: true,
        labels: ['Interacting', 'Moving independently']
    })*/
}

jsPsych.run(timeline);


// 'End' trial
timeline.push({
    type: htmlKeyboardResponse,
    stimulus: "<p style='text-align:center'>Congrats - you have completed your first experiment!  Press any key to end this trial.</p>"
});