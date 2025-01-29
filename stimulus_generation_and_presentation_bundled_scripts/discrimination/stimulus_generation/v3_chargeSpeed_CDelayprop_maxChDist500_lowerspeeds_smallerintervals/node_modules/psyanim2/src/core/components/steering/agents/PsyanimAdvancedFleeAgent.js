import PsyanimComponent from '../../../PsyanimComponent.js';

export default class PsyanimAdvancedFleeAgent extends PsyanimComponent {

    target;
    advancedFleeBehavior;
    vehicle;

    constructor(entity) {

        super(entity);
    }

    update(t, dt) {

        super.update(t, dt);

        if (this.target)
        {
            let steering = this.advancedFleeBehavior.getSteering(this.target);
    
            this.vehicle.steer(steering);    
        }
    }
}