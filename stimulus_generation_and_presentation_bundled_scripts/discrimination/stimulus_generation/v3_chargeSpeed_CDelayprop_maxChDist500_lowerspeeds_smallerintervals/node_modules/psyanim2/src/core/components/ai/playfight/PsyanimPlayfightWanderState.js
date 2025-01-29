import PsyanimFSMState from '../PsyanimFSMState.js';

import PsyanimWanderBehavior from '../../steering/PsyanimWanderBehavior.js';

import PsyanimPlayfightFSM from './PsyanimPlayfightFSM.js';

import PsyanimPlayfightChargeState from './PsyanimPlayfightChargeState.js';
import PsyanimPlayfightChargeDelayState from './PsyanimPlayfightChargeDelayState.js';
import PsyanimPlayfightFleeState from './PsyanimPlayfightFleeState.js';
import PsyanimVehicle from '../../steering/PsyanimVehicle.js';

import { PsyanimUtils, PsyanimDebug } from 'psyanim-utils';

export default class PsyanimPlayfightWanderState extends PsyanimFSMState {

    targetAgent;
    minTargetDistanceForCharge;
    maxTargetDistanceForCharge;

    breakDurationAverage; // ms
    breakDurationVariance; // ms

    fleeOrChargeWhenAttacked; // boolean
    panicDistance; // pixels
    fleeRate; // percentage

    minWanderDuration;

    constructor(fsm) {

        super(fsm);

        this.breakDurationAverage = 2000;
        this.breakDurationVariance = 1000;

        this.fleeOrChargeWhenAttacked = true;
        this.panicDistance = 300;
        this.fleeRate = 0.5;

        this.minTargetDistanceForCharge = 200;
        this.maxTargetDistanceForCharge = 500;

        this.minWanderDuration = 150;

        this._breakDuration = this.breakDurationAverage 
            + PsyanimUtils.getRandomInt(-this.breakDurationVariance, this.breakDurationVariance);

        // set default state variables
        this.fsm.setStateVariable('wanderTimer', 0);
        this.fsm.setStateVariable('distanceToTarget', this.maxTargetDistanceForCharge);
        this.fsm.setStateVariable('flee', false);
        this.fsm.setStateVariable('delayedCharge', false);

        this._targetAgentCharging = false;
        this._fleeOnPanic = false;

        /**
         *  Setup transitions here
         */

        this.addTransition(PsyanimPlayfightChargeState, 'wanderTimer', this._isChargeTransitionTriggered.bind(this));
        this.addTransition(PsyanimPlayfightFleeState, 'flee', (value) => value === true);
        this.addTransition(PsyanimPlayfightChargeDelayState, 'delayedCharge', (value) => value === true);
    }

    _recomputeFleeOnPanic() {

        let chargeRate = 1.0 - this.fleeRate;

        this._fleeOnPanic = Math.random() > chargeRate;
    }

    _handleTargetAgentStateEntered(state) {

        if (state === PsyanimPlayfightChargeState.name)
        {
            this._targetAgentCharging = true;

            this._recomputeFleeOnPanic();

            if (this.isActive && this.fsm.debugGraphics)
            {
                this.entity.color = 0xffff00;
            }
        }
    }

    _handleTargetAgentStateExited(state) {

        if (state === PsyanimPlayfightChargeState.name)
        {
            this._targetAgentCharging = false;

            this._fleeOnPanic = false;

            if (this.isActive && this.fsm.debugGraphics)
            {
                this.entity.color = 0x00ff00;
            }
        }
    }

    _isChargeTransitionTriggered(wanderTimer) {

        let distanceToTarget = this.fsm.getStateVariable('distanceToTarget');

        return (distanceToTarget < this.maxTargetDistanceForCharge)
            && (distanceToTarget > this.minTargetDistanceForCharge)
            && (wanderTimer > this._breakDuration);
    }

    afterCreate() {

        this._wanderBehavior = this.entity.getComponent(PsyanimWanderBehavior);
        this._vehicle = this.entity.getComponent(PsyanimVehicle);

        // subscribe to fsm events
        let targetAgentFSM = this.targetAgent.getComponent(PsyanimPlayfightFSM);

        targetAgentFSM.events.on('enter', this._handleTargetAgentStateEntered.bind(this));
        targetAgentFSM.events.on('exit', this._handleTargetAgentStateExited.bind(this));

        super.afterCreate();
    }

    onResume() {

        super.onResume();

        if (this.fsm.debugGraphics)
        {
            this.entity.color = 0x00ff00;
        }
    }

    enter() {

        super.enter();

        // compute a new break duration with random variance
        this._breakDuration = this.breakDurationAverage 
            + PsyanimUtils.getRandomInt(-this.breakDurationVariance, this.breakDurationVariance);

        this._recomputeFleeOnPanic();

        this.fsm.setStateVariable('wanderTimer', 0);
        this.fsm.setStateVariable('delayedCharge', false);
        this.fsm.setStateVariable('flee', false);

        if (this.fsm.debugGraphics)
        {
            this.entity.color = 0x00ff00;
        }
    }

    exit() {

        super.exit();
    }

    run(t, dt) {

        super.run(t, dt);

        // update state variables
        let updatedWanderTimer = this.fsm.getStateVariable('wanderTimer') + dt;
        
        let distanceToTarget = this.targetAgent.position
            .subtract(this.entity.position)
            .length();

        this.fsm.setStateVariable('distanceToTarget', distanceToTarget);

        if (distanceToTarget < this.maxTargetDistanceForCharge)
        {
            this.fsm.setStateVariable('wanderTimer', updatedWanderTimer);
        }

        if (this.fleeOrChargeWhenAttacked && this._targetAgentCharging)
        {
            if (distanceToTarget < this.panicDistance)
            {
                if (this._fleeOnPanic)
                {
                    this.fsm.setStateVariable('flee', true);
                }
                else if (updatedWanderTimer > this.minWanderDuration)
                {
                    this.fsm.setStateVariable('delayedCharge', true);
                }
            }
        }

        // apply steering
        let steering = this._wanderBehavior.getSteering();

        this._vehicle.steer(steering);
    }
}