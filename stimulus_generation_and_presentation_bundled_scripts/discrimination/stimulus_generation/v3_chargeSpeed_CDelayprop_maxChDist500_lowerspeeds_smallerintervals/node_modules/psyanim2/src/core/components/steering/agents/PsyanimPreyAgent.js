import PsyanimComponent from "../../../PsyanimComponent.js";

export default class PsyanimPredatorAgent extends PsyanimComponent {

    target;

    vehicle;

    preyBehavior;

    constructor(entity) {

        super(entity);
    }

    update(t, dt) {

        super.update(t, dt);

        // update steering params
        this.vehicle.maxSpeed = this.preyBehavior.maxSpeed;

        // compute steering
        let steering = this.preyBehavior.getSteering(this.target);

        this.vehicle.steer(steering);
    }
}