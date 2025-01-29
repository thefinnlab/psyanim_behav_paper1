import { initJsPsych } from 'jspsych';

import htmlKeyboardResponse from '@jspsych/plugin-html-keyboard-response';
import surveyText from '@jspsych/plugin-survey-text';
import surveyHtmlForm from '@jspsych/plugin-survey-html-form';
import HtmlSliderResponsePlugin from '@jspsych/plugin-html-slider-response';
import fullscreen from '@jspsych/plugin-fullscreen';
import surveyMultiChoice from '@jspsych/plugin-survey-multi-choice';
import surveyMultiSelect from '@jspsych/plugin-survey-multi-select';


import { 
    PsyanimApp, 
    PsyanimJsPsychPlugin,
    PsyanimJsPsychTrial,

    PsyanimFirebaseBrowserClient,

    PsyanimBasicPredatorBehavior,
    PsyanimMimic


} from 'psyanim2';

import firebaseJsonConfig from '../firebase.config.json';

import EmptyScene from './EmptyScene';
import PredatorPrey from './PredatorPrey';
import PredatorPreyMimic from './PredatorPreyMimic';

/**
 *  Handle user authentication and any other configuration
 */
const userID = 'demo_mimic';
const experimentName = 'defaultExperimentName';

/**
 *  Setup Psyanim App
 */
PsyanimApp.Instance.config.registerScene(EmptyScene);
PsyanimApp.Instance.config.registerScene(PredatorPreyMimic);

PsyanimApp.Instance.run();

PsyanimApp.Instance.setCanvasVisible(false);

/**
 *  Setup PsyanimJsPsychPlugin
 */
PsyanimJsPsychPlugin.setUserID(userID);
PsyanimJsPsychPlugin.setExperimentName(experimentName);

/**
 *  Setup jsPsych experiment
 */
const jsPsych = initJsPsych({
    //adapt_to_file_protocol: false,
    override_safe_mode: true,
    show_progress_bar: true,
});

let timeline = [];

// Full screen mode
timeline.push({
    type: fullscreen,
    fullscreen_mode: false
})

// 'welcome' trial
timeline.push({
    type: htmlKeyboardResponse,
    stimulus: 'Welcome to the experiment.  Press any key to begin.'
    //stimulus: '<div style="display: flex; justify-content: center; align-items: center; height: 100vh;">Welcome to the experiment.  Press any key to begin.</div>'
});

// setup predator-prey parameter-sets
let nTrials = 6;
let subtletyParams = [0, 30, 60, 90, 120, 150];

// we're going to randomly assign spawn locations to predator and prey for each trial
let leftSpawnPoint =  { x: 250, y: 300 };
let rightSpawnPoint = { x: 550, y: 300 };

let predatorSpawnPoints = [];
let preySpawnPoints = [];

let predatorColors = [];
let preyColors = [];

//0x000000: black; 0xcccccc: grey
for (let i = 0; i < 2; ++i)
{
    for (let j = 0; j < 2; ++j)
    {
        let predatorSpawnsOnLeft = i;
        let predatorColorBlack = j;

        if (predatorSpawnsOnLeft == 0 & predatorColorBlack == 0)
        {
            predatorSpawnPoints.push(leftSpawnPoint);
            predatorColors.push(0x000000);
            
            preySpawnPoints.push(rightSpawnPoint);
            preyColors.push(0xcccccc);
        }
        else if (predatorSpawnsOnLeft == 0 & predatorColorBlack == 1)
        {
            predatorSpawnPoints.push(leftSpawnPoint);
            predatorColors.push(0xcccccc);
            
            preySpawnPoints.push(rightSpawnPoint);
            preyColors.push(0x000000);
        }
        else if (predatorSpawnsOnLeft == 1 & predatorColorBlack == 0)
        {
            predatorSpawnPoints.push(rightSpawnPoint);
            predatorColors.push(0x000000);
            
            preySpawnPoints.push(leftSpawnPoint);
            preyColors.push(0xcccccc);
        }
        else{
            predatorSpawnPoints.push(rightSpawnPoint);
            predatorColors.push(0xcccccc);
            
            preySpawnPoints.push(leftSpawnPoint);
            preyColors.push(0x000000);
        }

    }
}  
// // Declear a shuffle function to randomize subtlety level
// const shuffle = (array) => { 
//     for (let i = array.length - 1; i > 0; i--) { 
//       const j = Math.floor(Math.random() * (i + 1)); 
//       [array[i], array[j]] = [array[j], array[i]]; 
//     } 
//     return array; 
//   }; 


/**
 *  Make chase stimuli
 */

// for (let j = 0; j < 4; ++j)
// {  
//     let subtletyParams_rand = shuffle(subtletyParams);
//     for (let i = 0; i < nTrials; ++i)
//     {
//         // modify the scene key to be something unique and register the scene
//         let sceneKey = PredatorPrey.key + '_subtlety_' + subtletyParams_rand[i] + '_pos' + predatorSpawnPoints[j].x + '_col' + predatorColors[j];
//         console.log(subtletyParams_rand);
//         let trial = new PsyanimJsPsychTrial(PredatorPrey, sceneKey);

//         // modify subtlety parameter for this scene
//         trial.setEntityParameter('predator', 'initialPosition', predatorSpawnPoints[j]);
//         trial.setEntityShapeParameter('predator', 'color', predatorColors[j]);
        
//         trial.setEntityParameter('prey', 'initialPosition', preySpawnPoints[j]);
//         trial.setEntityShapeParameter('prey', 'color', preyColors[j]);

//         trial.setPrefabParameter('predator', 'subtlety', subtletyParams_rand[i]);
        
//         trial.recordAnimationParameter('predator', PsyanimBasicPredatorBehavior,
//         '_subtletyAngle', 'subtletyAngle');

//         trial.addAgentNamesToRecord(['predator', 'prey']);
        

//         trial.duration = 6000;
//         trial.endTrialKeys = [];

//         timeline.push(trial.jsPsychTrialDefinition);
        
//    }
// }

/**
 *  Make invisible-chase (mimic) stimuli
 */
let nMimicTrials = 6;
let predatorInitialPosition = {x: 550, y: 300};
let mimicInitialPostion = {x: 250, y: 300};

// a few prey positions were generated and manually QCed as described in the preprint 
// let preyInitialPosition = {x: 726, y: 439};
// let preyInitialPosition = {x: 632, y: 580};
let preyInitialPosition = {x: 474, y: 422};
// let preyInitialPosition = {x: 609, y: 354};

// let preyInitialPosition = {x: 768, y: 67};
// let preyInitialPosition = {x: 750, y: 217};
// let preyInitialPosition = {x: 503, y: 102};
// let preyInitialPosition = {x: 417, y: 525};

// let preyInitialPosition = {x: 160, y: 331};
// let preyInitialPosition = {x: 554, y: 186};
// let preyInitialPosition = {x: 310, y: 414};
// let preyInitialPosition = {x: 628, y: 502};

// let preyInitialPosition = {x: 374, y: 95};
// let preyInitialPosition = {x: 293, y: 287};
// let preyInitialPosition = {x: 606, y: 142};
// let preyInitialPosition = {x: 54, y: 186};



for (let i = 0; i < nMimicTrials; ++i)
{

    let sceneKey = PredatorPreyMimic.key + '_mimic_' + subtletyParams[i] + '_preyPos' + preyInitialPosition.x + ', ' + preyInitialPosition.y + '_mimicPos' + mimicInitialPostion.x + ', ' + mimicInitialPostion.y; 
    let trial = new PsyanimJsPsychTrial(PredatorPreyMimic, sceneKey);

    trial.setEntityParameter('predator', 'initialPosition', predatorInitialPosition);
    trial.setEntityShapeParameter('predator', 'color', 0xcccccc);
        
    trial.setEntityParameter('prey', 'initialPosition', preyInitialPosition);
    //trial.setEntityShapeParameter('prey', 'color', preyColors[0]);
        
    trial.setPrefabParameter('predator', 'subtlety', subtletyParams[i]);
    
    // trial.setComponentParameter('preyMimic', PsyanimMimic, 'xOffset', mimicX);
    // trial.setComponentParameter('preyMimic', PsyanimMimic, 'yOffset', mimicY);
    trial.setEntityParameter('preyMimic', 'initialPosition', mimicInitialPostion);
    trial.setEntityShapeParameter('preyMimic', 'color', 0x000000);
    trial.setComponentParameter('preyMimic', PsyanimMimic, 'angleOffset', 180);

    trial.addAgentNamesToRecord(['predator', 'prey', 'preyMimic']);
    // Trial ends after 6s
    trial.duration = 6000;
    trial.endTrialKeys = [];

    trial.jsPsychData = {
        sceneKey: sceneKey
    };

    timeline.push(trial.jsPsychTrialDefinition);

    timeline.push({
                    type: htmlKeyboardResponse,
                    stimulus: 'Continue',
                    choices: ['enter']

         })


} 


// 'End' trial
timeline.push({
    type: htmlKeyboardResponse,
    stimulus: 'Congrats - you have completed your first experiment!  Press any key to end this trial.'
});

jsPsych.run(timeline);