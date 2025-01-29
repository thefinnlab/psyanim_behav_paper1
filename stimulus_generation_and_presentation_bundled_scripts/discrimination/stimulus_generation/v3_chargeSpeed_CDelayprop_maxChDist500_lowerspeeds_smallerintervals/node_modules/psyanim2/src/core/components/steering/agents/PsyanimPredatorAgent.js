import PsyanimComponent from '../../../PsyanimComponent.js';

export default class PsyanimPredatorAgent extends PsyanimComponent {

    target;

    vehicle;

    predatorBehavior;

    constructor(entity) {

        super(entity);
    }

    update(t, dt) {

        super.update(t, dt);

        // update steering params
        this.vehicle.maxSpeed = this.predatorBehavior.maxSpeed;

        // compute steering
        let steering = this.predatorBehavior.getSteering(this.target);

        this.vehicle.steer(steering);
    }
}