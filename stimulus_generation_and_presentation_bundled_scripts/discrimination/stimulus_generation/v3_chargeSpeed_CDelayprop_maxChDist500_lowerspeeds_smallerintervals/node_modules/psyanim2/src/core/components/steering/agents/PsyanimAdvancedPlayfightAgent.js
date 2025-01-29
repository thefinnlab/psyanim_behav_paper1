import PsyanimComponent from '../../../PsyanimComponent.js';

export default class PsyanimAdvancedPlayfightAgent extends PsyanimComponent {

    target;

    advancedPlayfightBehavior;
    vehicle;

    constructor(entity) {

        super(entity);
    }

    afterCreate() {

        if (this.target)
        {
            this.advancedPlayfightBehavior.setTarget(this.target);
        }
        else
        {
            console.error("playfight agent '" + this.entity.name + "' has no target assigned!");
        }
    }

    update(t, dt) {

        super.update(t, dt);

        // update steering params
        this.vehicle.maxSpeed = this.advancedPlayfightBehavior.maxSpeed;

        // compute steering
        this.advancedPlayfightBehavior.updateBreakTimer(dt);

        let steering = this.advancedPlayfightBehavior.getSteering();

        this.vehicle.steer(steering);
    }
}