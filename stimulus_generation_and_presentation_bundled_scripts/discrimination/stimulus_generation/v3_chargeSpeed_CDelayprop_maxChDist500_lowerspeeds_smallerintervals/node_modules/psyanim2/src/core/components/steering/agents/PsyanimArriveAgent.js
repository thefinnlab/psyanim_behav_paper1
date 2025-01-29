import PsyanimComponent from '../../../PsyanimComponent.js';

export default class PsyanimArriveAgent extends PsyanimComponent {

    target;
    arriveBehavior;
    vehicle;

    constructor(entity) {

        super(entity);
    }

    update(t, dt) {

        super.update(t, dt);

        if (this.target)
        {
            this.vehicle.maxSpeed = this.arriveBehavior.maxSpeed;

            let steering = this.arriveBehavior.getSteering(this.target);
    
            this.vehicle.steer(steering);    
        }
    }
}