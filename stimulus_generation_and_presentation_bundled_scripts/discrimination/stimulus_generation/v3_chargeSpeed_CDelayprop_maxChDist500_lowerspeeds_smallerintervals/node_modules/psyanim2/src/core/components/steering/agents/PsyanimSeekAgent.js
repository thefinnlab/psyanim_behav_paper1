import PsyanimComponent from '../../../PsyanimComponent.js';

export default class PsyanimSeekAgent extends PsyanimComponent {

    target;
    seekBehavior;
    vehicle;

    constructor(entity) {

        super(entity);
    }

    update(t, dt) {

        super.update(t, dt);

        this.vehicle.maxSpeed = this.seekBehavior.maxSpeed;

        let steering = this.seekBehavior.getSteering(this.target);

        this.vehicle.steer(steering);
    }
}