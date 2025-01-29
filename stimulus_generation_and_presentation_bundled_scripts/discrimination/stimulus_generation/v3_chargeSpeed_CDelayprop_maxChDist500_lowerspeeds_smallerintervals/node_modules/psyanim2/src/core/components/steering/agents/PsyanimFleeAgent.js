import PsyanimComponent from '../../../PsyanimComponent.js';

export default class PsyanimFleeAgent extends PsyanimComponent {

    target;
    fleeBehavior;
    vehicle;

    constructor(entity) {

        super(entity);
    }

    update(t, dt) {

        super.update(t, dt);

        this.vehicle.maxSpeed = this.fleeBehavior.maxSpeed;

        let steering = this.fleeBehavior.getSteering(this.target);

        this.vehicle.steer(steering);
    }
}