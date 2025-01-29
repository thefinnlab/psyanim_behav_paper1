import PsyanimConstants from '../../../PsyanimConstants.js';

import PsyanimVehicle from '../../steering/PsyanimVehicle.js';
import PsyanimArriveBehavior from '../../steering/PsyanimArriveBehavior.js';
import PsyanimSeekBehavior from '../../steering/PsyanimSeekBehavior.js';
import PsyanimWanderBehavior from '../../steering/PsyanimWanderBehavior.js';
import PsyanimFleeBehavior from '../../steering/PsyanimFleeBehavior.js';

import PsyanimFSM from '../PsyanimFSM.js';

import PsyanimPlayfightChargeState from './PsyanimPlayfightChargeState.js';
import PsyanimPlayfightChargeDelayState from './PsyanimPlayfightChargeDelayState.js';
import PsyanimPlayfightWanderState from './PsyanimPlayfightWanderState.js';
import PsyanimPlayfightFleeState from './PsyanimPlayfightFleeState.js';

import PsyanimSensor from '../../physics/PsyanimSensor.js';

export default class PsyanimPlayfightFSM extends PsyanimFSM {

    /** playfight params */
    target;

    /** wander state params */
    breakDurationAverage; 
    breakDurationVariance;
    minWanderDuration;

    minTargetDistanceForCharge;
    maxTargetDistanceForCharge;

    wanderFleeOrChargeWhenAttacked;
    wanderPanicDistance;
    wanderFleeRate;

    averageChargeDelay;
    chargeDelayVariance;

    /** flee state params */
    maxFleeDuration;

    /** charge state params */
    maxChargeDuration;

    /** arrive behavior params */
    maxChargeSpeed;
    maxChargeAcceleration;
    innerDecelerationRadius;
    outerDecelerationRadius;

    /** wander behavior params */
    maxWanderSpeed;
    maxWanderAcceleration;
    wanderRadius;
    wanderOffset;
    maxWanderAngleChangePerFrame;

    /** flee behavior params */
    maxFleeSpeed;
    maxFleeAcceleration;
    fleePanicDistance;

    constructor(entity) {

        super(entity);

        // default parameters
        this.debugLogging = false;

        // wander state
        this.breakDurationAverage = 2000;
        this.breakDurationVariance = 1000;
        this.minWanderDuration = 150;

        this.minTargetDistanceForCharge = 200;
        this.maxTargetDistanceForCharge = 500;

        this.wanderFleeOrChargeWhenAttacked = true;
        this.wanderPanicDistance = 250;
        this.wanderFleeRate = 0.5;

        this.averageChargeDelay = 600;
        this.chargeDelayVariance = 400;

        // flee state
        this.maxFleeDuration = 500;

        // charge state
        this.maxChargeDuration = 1500;

        // arrive behavior
        this.maxChargeSpeed = 9;
        this.maxChargeAcceleration = 0.4;

        this.innerDecelerationRadius = 12;
        this.outerDecelerationRadius = 30;

        // wander behavior
        this.maxWanderSpeed = 4;
        this.maxWanderAcceleration = 0.2;
        this.wanderRadius = 50;
        this.wanderOffset = 250;
        this.maxWanderAngleChangePerFrame = 20;

        // flee behavior
        this.maxFleeSpeed = 12;
        this.maxFleeAcceleration = 0.5;
        this.fleePanicDistance = 200;

        // attach components for this FSM
        this._sensor = this.entity.addComponent(PsyanimSensor);
        this._sensor.bodyShapeParams = {
            shapeType: PsyanimConstants.SHAPE_TYPE.CIRCLE,
            radius: this.entity.shapeParams.radius + 2
        };

        // this._sensor.debugLogging = true;

        this._sensor.events.on("triggerEnter", (entity) => {

            if (entity.name === this.target.name)
            {
                this.setStateVariable('chargeContact', true);
            }
        });

        // attach behaviors for this FSM
        this._vehicle = this.entity.addComponent(PsyanimVehicle);
        this._arriveBehavior = this.entity.addComponent(PsyanimArriveBehavior);
        this._seekBehavior = this.entity.addComponent(PsyanimSeekBehavior);
        this._wanderBehavior = this.entity.addComponent(PsyanimWanderBehavior);
        this._fleeBehavior = this.entity.addComponent(PsyanimFleeBehavior);
        
        // setup FSM
        this._chargeState = this.addState(PsyanimPlayfightChargeState);
        this._chargeDelayState = this.addState(PsyanimPlayfightChargeDelayState);
        this._wanderState = this.addState(PsyanimPlayfightWanderState);
        this._fleeState = this.addState(PsyanimPlayfightFleeState);

        this.initialState = this._wanderState;
    }

    onEnable() {

        super.onEnable();
    }

    onDisable() {

        super.onDisable();
    }

    onPause() {

        super.onPause();

        this.setStateVariable('chargeContact', false);
    }

    onStop() {

        super.onStop();

        this.setStateVariable('chargeContact', false);
    }

    onResume() {

        super.onResume();
    }

    afterCreate() {

        // charge state
        this._chargeState.sensor = this._sensor;
        this._chargeState.setTarget(this.target);
        this._chargeState.maxChargeDuration = this.maxChargeDuration;

        // charge delay state
        this._chargeDelayState.averageDelay = this.averageChargeDelay;
        this._chargeDelayState.delayVariance = this.chargeDelayVariance;

        // wander state
        this._wanderState.targetAgent = this.target;
        this._wanderState.breakDurationAverage = this.breakDurationAverage;
        this._wanderState.breakDurationVariance = this.breakDurationVariance;
        this._wanderState.minWanderDuration = this.minWanderDuration;
        this._wanderState.minTargetDistanceForCharge = this.minTargetDistanceForCharge;
        this._wanderState.maxTargetDistanceForCharge = this.maxTargetDistanceForCharge;
        this._wanderState.fleeOrChargeWhenAttacked = this.wanderFleeOrChargeWhenAttacked;
        this._wanderState.panicDistance = this.wanderPanicDistance;
        this._wanderState.fleeRate = this.wanderFleeRate;

        // flee state
        this._fleeState.target = this.target;
        this._fleeState.maxFleeDuration = this.maxFleeDuration;
        this._fleeState.maxSpeed = this.maxFleeSpeed;
        this._fleeState.maxAcceleration = this.maxFleeAcceleration;

        // arrive behavior
        this._arriveBehavior.maxSpeed = this.maxChargeSpeed;
        this._arriveBehavior.maxAcceleration = this.maxChargeAcceleration;
        this._arriveBehavior.innerDecelerationRadius = this.innerDecelerationRadius;
        this._arriveBehavior.outerDecelerationRadius = this.outerDecelerationRadius;

        // wander behavior
        this._wanderBehavior.seekBehavior = this._seekBehavior;
        this._wanderBehavior.maxSeekSpeed = this.maxWanderSpeed;
        this._wanderBehavior.maxSeekAcceleration = this.maxWanderAcceleration;
        this._wanderBehavior.radius = this.wanderRadius;
        this._wanderBehavior.offset = this.wanderOffset;
        this._wanderBehavior.maxWanderAngleChangePerFrame = this.maxWanderAngleChangePerFrame;

        // flee behavior
        this._fleeBehavior.maxSpeed = this.maxFleeSpeed;
        this._fleeBehavior.maxAcceleration = this.maxFleeAcceleration;
        this._fleeBehavior.panicDistance = this.fleePanicDistance;

        super.afterCreate();
    }

    beforeShutdown() {

        super.beforeShutdown();
    }

    update(t, dt) {
        
        super.update(t, dt);
    }
}