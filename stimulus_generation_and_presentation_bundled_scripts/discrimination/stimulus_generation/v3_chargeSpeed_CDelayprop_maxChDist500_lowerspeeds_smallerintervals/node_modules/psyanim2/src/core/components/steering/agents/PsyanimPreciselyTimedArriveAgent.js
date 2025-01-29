import PsyanimComponent from '../../../PsyanimComponent.js';

export default class PsyanimPreciselyTimedArriveAgent extends PsyanimComponent {

    target;
    advancedArriveBehavior;
    vehicle;

    constructor(entity) {
        
        super(entity);

        this.entity.disableFriction();
    }

    // this component is enabled at the start of any charge
    onEnable() {

        // at the start of charge, disable friction and recompute max speed
        this.entity.disableFriction();
        this.advancedArriveBehavior.computeMaxSpeed(this.target);
    }

    // this component is disabled at the end of any charge
    onDisable() {

        this.entity.enableFriction();
        this.entity.setVelocity(0, 0);
    }

    update(t, dt) {

        super.update(t, dt);

        // sync params from behaviors
        this.vehicle.maxSpeed = this.advancedArriveBehavior.maxSpeed;

        let steering = this.advancedArriveBehavior.getSteering(this.target);

        this.vehicle.steer(steering);
    }
}