import PsyanimApp from '../core/PsyanimApp.js';
import PsyanimComponent from '../core/PsyanimComponent.js';

export default class PsyanimJsPsychTrialLoader extends PsyanimComponent {

    trialInfo;
    documentReader;

    constructor(entity) {

        super(entity);
    }

    afterCreate() {

        super.afterCreate();

        this._getTrialMetadata();
    }

    _getTrialMetadata() {

        let trialIDs = this.trialInfo.map(info => info.trialID);

        this.documentReader.getTrialMetadataByIdAsync(trialIDs)
            .then((docs) => {

                this._trialMetadata = docs;

                this._getAnimationClips();
            })
            .catch((errorMessage) => {
                console.error(errorMessage);
            });
    }
    
    _getAnimationClips() {

        let clipIDs = [];

        this._trialMetadata
            .forEach(docData => {

                let doc = docData.data;

                let agentMetadata = doc.agentMetadata;

                agentMetadata.forEach(agent => {
                    
                    if (Object.hasOwn(agent, 'animationClipId') && 
                        !clipIDs.includes(agent.animationClipId))
                    {
                        clipIDs.push(agent.animationClipId);
                    }
                })
            });

        this.documentReader.getAnimationClipsByIdAsync(clipIDs)
            .then((clipData) => {

                this._animationData = clipData;

                this._saveDataAndEndTrial();
            })
            .catch((errorMessage) => {
                console.error(errorMessage);
            });
    }

    _saveDataAndEndTrial() {

        this.scene.registry.set('psyanim_trialMetadata', this._trialMetadata);
        this.scene.registry.set('psyanim_animationData', this._animationData);

        console.log('Successfully loaded experiment data!');

        // load the next trial
        PsyanimApp.Instance.events.emit('psyanim-jspsych-endTrial');
    }

    update(t, dt) {
        
        super.update(t, dt);
    }
}