import { v4 as uuidv4 } from 'uuid';

import { PsyanimApp } from 'psyanim2';

export default class PsyanimJsPsychDataWriterExtension {

    constructor(jsPsych) {
        this._jsPsych = jsPsych;
    }

    initialize(params) {

        this._userID = params.userID;
        this._experimentName = params.experimentName;
        this._documentWriter = params.documentWriter;
        
        this._documentID = uuidv4();
    }

    on_start(params) {
    }

    on_load(params) {
    }

    on_finish() {

        let jsPsychData = this._jsPsych.data.get().json();

        let data = {
            sessionID: PsyanimApp.Instance.sessionID,
            userID: this._userID,
            experimentName: this._experimentName,
            jsPsychData: jsPsychData,
        }

        this._documentWriter.addExperimentJsPsychData(data, this._documentID);

        return {};
    }
}

PsyanimJsPsychDataWriterExtension.info = {
    name: 'PsyanimJsPsychDataWriterExtension'
}