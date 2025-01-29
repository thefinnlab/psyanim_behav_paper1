import PsyanimComponent from '../../PsyanimComponent.js';

import {
    PsyanimDebug
} from 'psyanim-utils';

export default class PsyanimPlayfightBehavior extends PsyanimComponent {

    breakDuration;

    fleeBehavior;
    arriveBehavior;
    wanderBehavior;

    debug;

    constructor(entity) {

        super(entity);

        this.debug = false;

        this.breakDuration = 1500;

        this._breakTimer = 0;

        this._setState(PsyanimPlayfightBehavior.STATE.WANDERING);
    }

    handleCollision(matterCollisionData) {

        if (this.debug)
        {
            PsyanimDebug.log("playfight collision occured at t = " + this.entity.scene.time.now / 1000);
        }

        this._breakTimer = 0;

        this._setState(PsyanimPlayfightBehavior.STATE.FLEEING);    
    }

    get maxSpeed() {

        switch (this._state) {

            case PsyanimPlayfightBehavior.STATE.CHARGING:

                return this.arriveBehavior.maxSpeed;

            case PsyanimPlayfightBehavior.STATE.WANDERING:

                return this.wanderBehavior.seekBehavior.maxSpeed;

            case PsyanimPlayfightBehavior.STATE.FLEEING:

                return this.fleeBehavior.maxSpeed;
        }
    }

    get maxAcceleration() {

        switch (this._state) {

            case PsyanimPlayfightBehavior.STATE.CHARGING:

                return this.arriveBehavior.maxAcceleration;

            case PsyanimPlayfightBehavior.STATE.WANDERING:

                return this.wanderBehavior.seekBehavior.maxAcceleration;

            case PsyanimPlayfightBehavior.STATE.FLEEING:

                return this.fleeBehavior.maxAcceleration;
        }
    }

    _setState(state) {

        this._state = state;

        switch (this._state) {

            case PsyanimPlayfightBehavior.STATE.CHARGING:

                if (this.debug)
                {
                    PsyanimDebug.log(this.entity.name + ' entered playfight state: CHARGING');
                }

                break;

            case PsyanimPlayfightBehavior.STATE.WANDERING:

                if (this.debug)
                {
                    PsyanimDebug.log(this.entity.name + ' entered playfight state: WANDERING');
                }

                break;

            case PsyanimPlayfightBehavior.STATE.FLEEING:

                if (this.debug)
                {
                    PsyanimDebug.log(this.entity.name + ' entered playfight state: FLEEING');
                }

                break;
        }
    }

    updateBreakTimer(dt) {

        if (this._state == PsyanimPlayfightBehavior.STATE.WANDERING)
        {
            this._breakTimer += dt;

            if (this._breakTimer >= this.breakDuration)
            {
                this._setState(PsyanimPlayfightBehavior.STATE.CHARGING);
            }
        }
        else if (this._state == PsyanimPlayfightBehavior.STATE.FLEEING)
        {
            this._breakTimer += dt;
        }
    }

    getSteering(target) {

        let distanceToTarget = 0;

        switch (this._state) {

            case PsyanimPlayfightBehavior.STATE.CHARGING:

                return this.arriveBehavior.getSteering(target);

            case PsyanimPlayfightBehavior.STATE.WANDERING:

                distanceToTarget = this.entity.position
                    .subtract(target.position)
                    .length();

                if (distanceToTarget < this.fleeBehavior.panicDistance)
                {
                    this._setState(PsyanimPlayfightBehavior.STATE.FLEEING);

                    return this.fleeBehavior.getSteering(target);
                }

                return this.wanderBehavior.getSteering();

            case PsyanimPlayfightBehavior.STATE.FLEEING:

                distanceToTarget = this.entity.position
                    .subtract(target.position)
                    .length();

                if (distanceToTarget > this.fleeBehavior.panicDistance)
                {
                    this._setState(PsyanimPlayfightBehavior.STATE.WANDERING);

                    return this.wanderBehavior.getSteering();
                }

                return this.fleeBehavior.getSteering(target);
        }
    }
}

PsyanimPlayfightBehavior.STATE = {
    WANDERING: 0x0001,
    CHARGING: 0x0002,
    FLEEING: 0x0004
};