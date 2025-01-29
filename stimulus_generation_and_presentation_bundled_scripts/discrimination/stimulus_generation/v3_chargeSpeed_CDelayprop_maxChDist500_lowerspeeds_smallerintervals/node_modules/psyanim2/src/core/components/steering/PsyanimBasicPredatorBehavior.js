import PsyanimComponent from "../../PsyanimComponent.js";

import {
    PsyanimDebug
} from 'psyanim-utils';

export default class PsyanimBasicPredatorBehavior extends PsyanimComponent {

    /**
     *  Basic algorithm is to wander until a target is in sight, and then pursue that target, but
     *  more subtly by varying the pursuit angle a tad bit off from the normal direction an 
     *  'arrive' behavior would approach at.
     */

    subtlety; // 'degrees'
    subtletyLag; // 'ms'

    boredomDistance; // max. distance in 'px' to target beyond which this predator will no longer chase

    fovSensor;

    arriveBehavior;
    wanderBehavior;
    
    minimumWanderTime;

    debug;

    constructor(entity) {

        super(entity);

        this.debug = false;

        this.subtlety = 30; // 'degrees'
        this.subtletyLag = 500; // 'ms'
    
        this.minimumWanderTime = 1000;

        this._state = PsyanimBasicPredatorBehavior.STATE.WANDERING;

        this._arriveTarget = this.scene.addEntity(this.entity.name + '_arriveTarget');

        this._subtletyUpdateTimer = 0;

        this._wanderTimer = 0;
    }

    afterCreate() {

        super.afterCreate();

        // use an initial angle and subtlety that's randomized
        this.entity.setAngle(360 * (2 * Math.random() - 1));
        this._recomputeSubtletyAngle();
    }

    get maxSpeed() {

        switch(this._state) {

            case PsyanimBasicPredatorBehavior.STATE.WANDERING:

                return this.wanderBehavior.seekBehavior.maxSpeed;

            case PsyanimBasicPredatorBehavior.STATE.PURSUING:

                return this.arriveBehavior.maxSpeed;

            default:

                console.error("ERROR: invalid state: " + this._state);
                return 0;
        }
    }

    getSteering(target) {

        // determine if target is in sight
        let targetInSight = false;

        if (this.fovSensor)
        {
            let entitiesInSight = this.fovSensor.getEntitiesInSight([target]);

            if (entitiesInSight.length != 0)
            {
                if (entitiesInSight.includes(target))
                {
                    targetInSight = true;
                }
            }    
        }
        else // treat it as 360 degree FOV
        {
            targetInSight = true;
        }

        // update state
        if (targetInSight && this._wanderTimer > this.minimumWanderTime)
        {
            if (this._state != PsyanimBasicPredatorBehavior.STATE.PURSUING)
            {
                if (this.debug)
                {
                    PsyanimDebug.log("Predator '" + this.entity.name + "' state = PURSUING.");
                }
    
                this._state = PsyanimBasicPredatorBehavior.STATE.PURSUING;    
            }
        }
        else
        {
            if (this._state != PsyanimBasicPredatorBehavior.STATE.WANDERING)
            {
                if (this.debug)
                {
                    PsyanimDebug.log("Predator '" + this.entity.name + "' state = WANDERING.");
                }

                this._state = PsyanimBasicPredatorBehavior.STATE.WANDERING;
                this._wanderTimer = 0;
            }
        }

        // compute steering
        if (this._state == PsyanimBasicPredatorBehavior.STATE.PURSUING)
        {
            let targetRelativePosition = target.position
                .subtract(this.entity.position);

            targetRelativePosition.rotate(this._subtletyAngle * Math.PI / 180.0);

            let newTargetPosition = this.entity.position
                .add(targetRelativePosition);

            this._arriveTarget.position = newTargetPosition;

            return this.arriveBehavior.getSteering(this._arriveTarget);
        }
        else if (this._state == PsyanimBasicPredatorBehavior.STATE.WANDERING)
        {
            return this.wanderBehavior.getSteering();
        }
    }

    _recomputeSubtletyAngle() {

        this._subtletyAngle = this.subtlety * (2 * Math.random() - 1);
    }

    update(t, dt) {
     
        super.update(t, dt);

        if (this.fovSensor)
        {
            this.fovSensor.fovRange = this.boredomDistance;
        }

        this._wanderTimer += dt;

        this._subtletyUpdateTimer += dt;

        if (this._subtletyUpdateTimer > this.subtletyLag)
        {
            this._subtletyUpdateTimer = 0;

            if (this._state == PsyanimBasicPredatorBehavior.STATE.PURSUING)
            {
                this._recomputeSubtletyAngle();
            }
        }
    }
}

PsyanimBasicPredatorBehavior.STATE = {
    WANDERING: 0x0001,
    PURSUING: 0x0002,
};