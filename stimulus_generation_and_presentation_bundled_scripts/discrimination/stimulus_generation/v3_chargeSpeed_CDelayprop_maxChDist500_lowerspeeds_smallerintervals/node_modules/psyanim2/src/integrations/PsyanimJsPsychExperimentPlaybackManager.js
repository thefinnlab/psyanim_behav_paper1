import PsyanimComponent from '../core/PsyanimComponent.js';
import PsyanimApp from '../core/PsyanimApp.js';

export default class PsyanimJsPsychExperimentPlaybackManager extends PsyanimComponent {

    trialSelector;
    experimentPlayer;

    constructor(entity) {

        super(entity);
    }

    afterCreate() {

        super.afterCreate();

        let nextTrialData = this.trialSelector.getTrialData();

        this.experimentPlayer.events.on('playbackComplete', 
            this._handlePlaybackComplete.bind(this));

        this.experimentPlayer.loadExperiment(nextTrialData);
    }

    _handlePlaybackComplete() {

        PsyanimApp.Instance.events.emit('playbackComplete');
    }

    update(t, dt) {
        
        super.update(t, dt);
    }
}