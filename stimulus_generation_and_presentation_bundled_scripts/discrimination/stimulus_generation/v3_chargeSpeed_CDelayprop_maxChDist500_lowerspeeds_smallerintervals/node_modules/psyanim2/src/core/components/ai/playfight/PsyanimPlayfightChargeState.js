import PsyanimFSMState from '../PsyanimFSMState.js';

import { PsyanimDebug } from 'psyanim-utils';

import PsyanimVehicle from '../../steering/PsyanimVehicle.js';
import PsyanimArriveBehavior from '../../steering/PsyanimArriveBehavior.js';

import PsyanimPlayfightWanderState from './PsyanimPlayfightWanderState.js'

export default class PsyanimPlayfightChargeState extends PsyanimFSMState {

    sensor;

    maxChargeDuration;

    constructor(fsm) {

        super(fsm);

        this.maxChargeDuration = 1500;

        this.fsm.setStateVariable('charge', false);
        this.fsm.setStateVariable('chargeTimer', 0);

        /**
         *  Setup transitions here
         */

        this.addTransition(PsyanimPlayfightWanderState, 'charge', (value) => value === false);
    }

    afterCreate() {

    }

    onPause() {

        super.onPause();
    }

    onStop() {

        super.onStop();
    }

    onResume() {

        super.onResume();

        if (this.fsm.debugGraphics)
        {
            this.entity.color = 0xff0000;
        }
    }

    setTarget(target) {

        if (target) 
        {
            this.sensor.events.on('triggerEnter', this._handleCollision.bind(this));

            this._target = target;
        }
        else
        {
            PsyanimDebug.error('target field is null!');
        }
    }

    _handleCollision(entity) {

        if  (entity.name === this._target.name)
        {
            this.fsm.setStateVariable('charge', false);
        }
    }

    enter() {

        super.enter();

        this._vehicle = this.entity.getComponent(PsyanimVehicle);
        this._arriveBehavior = this.entity.getComponent(PsyanimArriveBehavior);

        if (this.sensor.isIntersecting(this._target))
        {
            this.fsm.setStateVariable('charge', false);
        }
        else
        {
            this.fsm.setStateVariable('charge', true);
        }

        this.fsm.setStateVariable('chargeTimer', 0);

        if (this.fsm.debugGraphics)
        {
            this.entity.color = 0xff0000;
        }
    }

    exit() {

        super.exit();
    }

    run(t, dt) {

        super.run();

        // update charge timer and see if we've exceeded the max charge duration
        let updatedChargeTimer = this.fsm.getStateVariable('chargeTimer') + dt;

        this.fsm.setStateVariable('chargeTimer', updatedChargeTimer);

        if (updatedChargeTimer > this.maxChargeDuration)
        {
            this.fsm.setStateVariable('charge', false);
        }

        // apply steering
        let steering = this._arriveBehavior.getSteering(this._target);

        this._vehicle.steer(steering);
    }
}