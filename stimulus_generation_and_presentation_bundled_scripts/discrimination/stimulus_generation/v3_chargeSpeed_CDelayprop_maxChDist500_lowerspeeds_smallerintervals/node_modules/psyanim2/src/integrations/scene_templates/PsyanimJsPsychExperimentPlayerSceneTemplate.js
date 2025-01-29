import PsyanimJsPsychExperimentPlayer from '../PsyanimJsPsychExperimentPlayer.js';
import PsyanimJsPsychTrialSelector from '../PsyanimJsPsychTrialSelector.js';
import PsyanimJsPsychExperimentPlaybackManager from '../PsyanimJsPsychExperimentPlaybackManager.js';

export default {
    key: 'Psyanim jsPsych Experiment Player Main Scene',
    wrapScreenBoundary: false,
    entities: [
        {
            name: 'playbackManager',
            components: [
                { 
                    type: PsyanimJsPsychExperimentPlaybackManager,
                    params: {
                        trialSelector: {
                            entityName: 'trialSelector',
                            componentType: PsyanimJsPsychTrialSelector
                        },
                        experimentPlayer: {
                            entityName: 'experimentPlayer',
                            componentType: PsyanimJsPsychExperimentPlayer
                        }
                    }
                }
            ]
        },
        {
            name: 'trialSelector',
            components: [
                { type: PsyanimJsPsychTrialSelector }
            ]
        },
        {
            name: 'experimentPlayer',
            components: [
                { type: PsyanimJsPsychExperimentPlayer }
            ]
        }
    ]
}