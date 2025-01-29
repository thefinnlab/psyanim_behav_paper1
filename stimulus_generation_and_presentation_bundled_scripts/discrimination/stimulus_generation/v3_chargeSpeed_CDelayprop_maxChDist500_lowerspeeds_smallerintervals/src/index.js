import { initJsPsych } from 'jspsych';

import htmlKeyboardResponse from '@jspsych/plugin-html-keyboard-response';

import {
    PsyanimApp,
    PsyanimJsPsychPlugin,
    PsyanimJsPsychTrial,
    PsyanimFirebaseBrowserClient, //Firebase uncomment
    PsyanimJsPsychDataWriterExtension, //Firebase uncomment 
    PsyanimPlayfightFSM,
    PsyanimPlayfightHFSM
} from 'psyanim2';

import firebaseJsonConfig from '../firebase.config.json'; //Firebase uncomment

import playfightScene from './playfight.js';
/**
 *  Handle user authentication and any other configuration
 */

const chargeDelayVariance = 50;
const maxTargetDistanceForCharge = 500; //450 backup, original 500
const maxSeparationDuration = 1000; // 750 backup, original 1000 //100;
const userID = 'stimGen_playfight_v3_set18'; //'stimGeneration_playfight_v2_higherAcc'; //'stimGeneration_playfight_v2';
const experimentName = 'playFight_slower_BD2k_var200_CDelayvar' + chargeDelayVariance + '_maxChDist' + maxTargetDistanceForCharge + '_maxSepDur' + maxSeparationDuration;
// e.g. 'playFight_slower_BD2k_var200_CD500_var50_maxChDist500'

/**
 *  Setup Psyanim App
 */

PsyanimApp.Instance.run();

PsyanimApp.Instance.setCanvasVisible(true);
//PsyanimApp.Instance.config.phaserConfig.width = 400;
//PsyanimApp.Instance.config.phaserConfig.height = 300;
/**
 *  Setup PsyanimJsPsychPlugin
 */
PsyanimJsPsychPlugin.setUserID(userID);
PsyanimJsPsychPlugin.setExperimentName(experimentName);

const firebaseClient = new PsyanimFirebaseBrowserClient(firebaseJsonConfig); //Firebase uncomment
PsyanimJsPsychPlugin.setDocumentWriter(firebaseClient);

/**
 *  Setup jsPsych experiment 
 */
const jsPsych = initJsPsych({

    // Firebase uncomment
    extensions: [{
        type: PsyanimJsPsychDataWriterExtension,
        params: {
            documentWriter: firebaseClient,
            userID: userID,
            experimentName: experimentName
        }
    }],
});

let timeline = [];
// 'Welcome' trial
timeline.push({
    type: htmlKeyboardResponse,
    stimulus: "<p style='text-align:center'>Welcome to the experiment.  Press any key to begin.</p>",
});

let nRep = 5; //5; // 1st version (feb 2) had 5 //need 10 of each color comb in the end  (not making more when testing diff versions)
let nTrialsPerLevel = nRep * 2; // 2 unique combinations.
let colors = { black: 0x000000, grey: 0xcccccc }; //0x000000: black; 0xcccccc: grey
let agents = ['agent1', 'agent2'];
let chargeSpeeds = [1.5, 2.75, 4, 5.25, 6.5, 7.75, 9];
let chargeAcc = [0.225, 0.4125, 0.6, 0.7875, 0.975, 1.1625, 1.35]; //15% of speed
let agent1Color, agent2Color;
//let itertxt; 
for (let i = 0; i < chargeSpeeds.length; ++i) { //chargeSpeeds.length
    console.log(i)
    if (i === 0) { //set 7,8,9,19,11,12,13,14,15,16,17,18
        //if (i === 0 || i === 4 || i === 6) { //set 5, 6
        for (let iter = 0; iter < nTrialsPerLevel; ++iter) { //nTrialsPerLevel
            if (iter < nTrialsPerLevel / 2) {
                agent1Color = 'black';
                agent2Color = 'grey';
            } else {
                agent1Color = 'grey';
                agent2Color = 'black';
            }
            let avgCDelay = Math.round(100 + 500 * (1 - ((15 / 100) * i))); //150-600
            //chSp3_ag1Lblack_rep1_BD2k_var200_CD500_var50_maxChD500
            let sceneKey = 'chSp' + chargeSpeeds[i] + '_ag1L' + agent1Color + '_rep' + iter + '_avgCDelay' + avgCDelay; // backup 450
            let playfightSceneTrial = new PsyanimJsPsychTrial(playfightScene, sceneKey);
            playfightSceneTrial.duration = 8000; //8000;
            playfightSceneTrial.setEntityShapeParameter('agent1', 'color', colors[agent1Color]);
            playfightSceneTrial.setEntityShapeParameter('agent2', 'color', colors[agent2Color]);
            for (let iagent = 0; iagent < 2; ++iagent) {
                //console.log(iagent)
                let agent = agents[iagent] // 'agent1'
                playfightSceneTrial.setComponentParameter(agent, PsyanimPlayfightFSM, 'maxChargeSpeed', chargeSpeeds[i]); //speed[i]);
                playfightSceneTrial.setComponentParameter(agent, PsyanimPlayfightFSM, 'maxChargeAcceleration', chargeAcc[i]); // .1 * chargeSpeeds[i]); //.2 + chargeAcc[i]);

                playfightSceneTrial.setComponentParameter(agent, PsyanimPlayfightFSM, 'averageChargeDelay', avgCDelay);
                playfightSceneTrial.setComponentParameter(agent, PsyanimPlayfightFSM, 'chargeDelayVariance', chargeDelayVariance);

                playfightSceneTrial.setComponentParameter(agent, PsyanimPlayfightFSM, 'maxTargetDistanceForCharge', maxTargetDistanceForCharge);

                playfightSceneTrial.setComponentParameter(agent, PsyanimPlayfightHFSM, 'maxSeparationDuration', maxSeparationDuration);


                // averageChargeDelay = 200;  chargeDelayVariance = 100; 
            }
            playfightSceneTrial.subtext = '<strong> chargeSpeed</strong>:' + chargeSpeeds[i] + ',<strong> chargeAcc</strong>:' + chargeAcc[i] + ',<strong> chargeDelay</strong>:' + avgCDelay + '<br>' + sceneKey;
            playfightSceneTrial.endTrialKeys = [];
            playfightSceneTrial.endTrialOnPlaybackComplete = true;
            playfightSceneTrial.addAgentNamesToRecord(['agent1', 'agent2']);
            playfightSceneTrial.recordStateLogs = true;
            timeline.push(playfightSceneTrial.jsPsychTrialDefinition);
        }
    }
}

// 'End' trial
timeline.push({
    type: htmlKeyboardResponse,
    stimulus: "<p style='text-align:center'>Congrats - you have completed your first experiment!  Press any key to end this trial.</p>",
    // extensions: [
    //     { type: PsyanimJsPsychDataWriterExtension }
    // ],
});

jsPsych.run(timeline);