import PsyanimComponent from '../../PsyanimComponent.js';

import {
    PsyanimDebug
} from 'psyanim-utils';

export default class PsyanimAdvancedPlayfightBehavior extends PsyanimComponent {

    collisionFrequency;
    breakDuration;

    fleeBehavior;
    advancedArriveBehavior;
    wanderBehavior;

    debug;

    constructor(entity) {

        super(entity);

        this.debug = false;

        // TODO: can we just change 'collision frequency' & 'break duration' to seconds instead of ms?
        this.collisionFrequency = 2000;
        this.breakDuration = 1500;
    
        this._breakTimer = 0;

        this._setState(PsyanimAdvancedPlayfightBehavior.STATE.WANDERING);

        this._arriveTarget = this.entity.scene
            .addEntity(this.entity.name + '_arriveTarget', 0, 0, { isEmpty: true });
    }

    onEnable() {

        this.entity.disableFriction();
    }

    onDisable() {

        this.entity.enableFriction();
    }

    setTarget(target) {

        this._target = target;

        this.entity.setOnCollideWith(this._target.body, (matterCollisionData) => {
            this.handleCollision(matterCollisionData) 
        });
    }

    handleCollision(matterCollisionData) {

        if (this.debug)
        {
            PsyanimDebug.log("playfight collision occured at t = " + this.entity.scene.time.now / 1000);
        }

        this._breakTimer = 0;

        this._setState(PsyanimAdvancedPlayfightBehavior.STATE.FLEEING);    
    }

    get maxSpeed() {

        switch (this._state) {

            case PsyanimAdvancedPlayfightBehavior.STATE.CHARGING:

                return this.advancedArriveBehavior.maxSpeed;

            case PsyanimAdvancedPlayfightBehavior.STATE.WANDERING:

                return this.wanderBehavior.seekBehavior.maxSpeed;

            case PsyanimAdvancedPlayfightBehavior.STATE.FLEEING:

                return this.fleeBehavior.maxSpeed;
        }
    }

    get maxAcceleration() {

        switch (this._state) {

            case PsyanimAdvancedPlayfightBehavior.STATE.CHARGING:

                return this.advancedArriveBehavior.maxAcceleration;

            case PsyanimAdvancedPlayfightBehavior.STATE.WANDERING:

                return this.wanderBehavior.seekBehavior.maxAcceleration;

            case PsyanimAdvancedPlayfightBehavior.STATE.FLEEING:

                return this.fleeBehavior.maxAcceleration;
            }
    }

    _setState(state) {

        this._state = state;

        switch (this._state) {

            case PsyanimAdvancedPlayfightBehavior.STATE.CHARGING:

                if (this.debug)
                {
                    PsyanimDebug.log(this.entity.name + ' entered playfight state: CHARGING');
                }

                this.entity.disableFriction();

                // move the arrive target to the midpoint between this.entity and this._target
                this._arriveTarget.position = this._target.position
                    .subtract(this.entity.position)
                    .scale(0.5)
                    .add(this.entity.position);

                // compute charge duration necessary to satisfy collision frequency
                this.advancedArriveBehavior.chargeDuration = (this.collisionFrequency - this._breakTimer) / 1000;

                if (this.debug)
                {
                    PsyanimDebug.log("charge duration = " + this.advancedArriveBehavior.chargeDuration);
                }

                // recompute max speed based on distance to target at start of charge
                this.advancedArriveBehavior.computeMaxSpeed(this._arriveTarget);

                break;

            case PsyanimAdvancedPlayfightBehavior.STATE.WANDERING:

                if (this.debug)
                {
                    PsyanimDebug.log(this.entity.name + ' entered playfight state: WANDERING');
                }

                this.entity.enableFriction();

                break;

            case PsyanimAdvancedPlayfightBehavior.STATE.FLEEING:

                if (this.debug)
                {
                    PsyanimDebug.log(this.entity.name + ' entered playfight state: FLEEING');
                }

                this.entity.enableFriction();

                break;
        }
    }

    updateBreakTimer(dt) {

        if (this._state == PsyanimAdvancedPlayfightBehavior.STATE.WANDERING)
        {
            this._breakTimer += dt;

            if (this._breakTimer >= this.breakDuration)
            {
                this._setState(PsyanimAdvancedPlayfightBehavior.STATE.CHARGING);
            }
        }
        else if (this._state == PsyanimAdvancedPlayfightBehavior.STATE.FLEEING)
        {
            this._breakTimer += dt;
        }
    }

    getSteering() {

        let distanceToTarget = 0;

        switch (this._state) {

            case PsyanimAdvancedPlayfightBehavior.STATE.CHARGING:

                return this.advancedArriveBehavior.getSteering(this._target);

            case PsyanimAdvancedPlayfightBehavior.STATE.WANDERING:

                distanceToTarget = this.entity.position
                    .subtract(this._target.position)
                    .length();

                if (distanceToTarget < this.fleeBehavior.panicDistance)
                {
                    // transition to fleeing and fall through to the next case
                    this._setState(PsyanimAdvancedPlayfightBehavior.STATE.FLEEING);
                }
                else
                {
                    return this.wanderBehavior.getSteering();
                }

            case PsyanimAdvancedPlayfightBehavior.STATE.FLEEING:

                distanceToTarget = this.entity.position
                    .subtract(this._target.position)
                    .length();

                if (distanceToTarget > this.fleeBehavior.panicDistance)
                {
                    this._setState(PsyanimAdvancedPlayfightBehavior.STATE.WANDERING);
                }

                return this.fleeBehavior.getSteering(this._target);
        }
    }
}

PsyanimAdvancedPlayfightBehavior.STATE = {
    WANDERING: 0x0001,
    CHARGING: 0x0002,
    FLEEING: 0x0004
};