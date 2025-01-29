import PsyanimBasicHFSM from "../PsyanimBasicHFSM.js";

import PsyanimPlayfightFSM from './PsyanimPlayfightFSM.js';
import PsyanimPlayfightSeparationFSM from './PsyanimPlayfightSeparationFSM.js';

import PsyanimSensor from "../../physics/PsyanimSensor.js";

export default class PsyanimPlayfightHFSM extends PsyanimBasicHFSM {

    playfightFSM;
    separationFSM;

    maxSeparationDuration;

    constructor(entity) {

        super(entity);

        this.maxSeparationDuration = 100;
    }
    
    afterCreate() {

        super.afterCreate();

        // add sub-state machines
        this.addSubStateMachine(this.playfightFSM);
        this.addSubStateMachine(this.separationFSM);

        this.playfightFSM.debugLogging = this.debugLogging;
        this.separationFSM.debugLogging = this.debugLogging;

        this.playfightFSM.debugGraphics = this.debugGraphics;
        this.separationFSM.debugGraphics = this.debugGraphics;

        // add interrupts
        this.addInterrupt(PsyanimPlayfightFSM, 'chargeContact', 
            value => value === true, 
            PsyanimPlayfightSeparationFSM);

        this.addInterrupt(PsyanimPlayfightSeparationFSM, 'separationTimer', 
            value => value >= this.maxSeparationDuration);

        // setup initial substate machine to run
        this.initialSubStateMachine = this.playfightFSM;

        this._sensor = this.entity.getComponent(PsyanimSensor);
    }

    update(t, dt) {

        super.update(t, dt);
    }
}