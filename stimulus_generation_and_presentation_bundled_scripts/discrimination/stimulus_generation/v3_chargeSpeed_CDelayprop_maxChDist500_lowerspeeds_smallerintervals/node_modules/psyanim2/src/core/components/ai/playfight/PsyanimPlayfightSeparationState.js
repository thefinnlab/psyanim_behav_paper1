import PsyanimFSMState from '../PsyanimFSMState.js';

import PsyanimVehicle from '../../steering/PsyanimVehicle.js';
import PsyanimFleeBehavior from '../../steering/PsyanimFleeBehavior.js';

export default class PsyanimPlayfightSeparationState extends PsyanimFSMState {

    target;

    maxSpeed;
    maxAcceleration;

    // TODO: let's add a separation flee subletly here too, so they don't necessarily flee in exact
    // opposite directions...
    
    constructor(fsm) {

        super(fsm);

        this.maxSpeed = 12;
        this.maxAcceleration = 0.5;

        // setup state variables
        this.fsm.setStateVariable('separationTimer', 0);
    }

    afterCreate() {

        super.afterCreate();

        this._vehicle = this.entity.getComponent(PsyanimVehicle);
        this._fleeBehavior = this.entity.getComponent(PsyanimFleeBehavior);
    }

    onPause() {

        super.onPause();

        this.fsm.setStateVariable('separationTimer', 0);
    }

    onStop() {

        super.onStop();

        this.fsm.setStateVariable('separationTimer', 0);
    }

    onResume() {

        super.onResume();

        if (this.fsm.debugGraphics)
        {
            this.entity.color = 0x800080;
        }
    }

    enter() {

        super.enter();

        if (this.fsm.debugGraphics)
        {
            this.entity.color = 0x800080;
        }

        this._fleeBehavior.maxSpeed = this.maxSpeed;
        this._fleeBehavior.maxAcceleration = this.maxAcceleration;

        this.fsm.setStateVariable('separationTimer', 0);
    }

    exit() {

        super.exit();
    }

    run(t, dt) {

        super.run(t, dt);

        // update separation timer
        let separationDuration = this.fsm.getStateVariable('separationTimer');

        this.fsm.setStateVariable('separationTimer', separationDuration + dt);

        // apply steering force
        let steering = this._fleeBehavior.getSteering(this.target);

        this._vehicle.steer(steering);
    }
}