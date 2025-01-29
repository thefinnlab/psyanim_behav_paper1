import PsyanimJsPsychTrialLoader from '../PsyanimJsPsychTrialLoader.js';

export default {
    key: 'Psyanim JsPsych Experiment Loading Scene',
    entities: [
        {
            name: 'experimentLoader',
            components: [
                { type: PsyanimJsPsychTrialLoader }
            ]
        }
    ]
}