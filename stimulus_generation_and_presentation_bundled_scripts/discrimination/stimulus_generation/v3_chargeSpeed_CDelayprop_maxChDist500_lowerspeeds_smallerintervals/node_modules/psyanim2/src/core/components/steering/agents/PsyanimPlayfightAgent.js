import PsyanimComponent from '../../../PsyanimComponent.js';

export default class PsyanimPlayfightAgent extends PsyanimComponent {

    target;

    playfightBehavior;
    vehicle;

    constructor(entity) {

        super(entity);
    }

    afterCreate() {

        if (this.target)
        {
            this.entity.setOnCollideWith(this.target.body, (matterCollisionData) => {
                this.playfightBehavior.handleCollision(matterCollisionData) 
            });
        }
    }

    update(t, dt) {

        super.update(t, dt);

        // update steering params
        this.vehicle.maxSpeed = this.playfightBehavior.maxSpeed;

        // compute steering
        this.playfightBehavior.updateBreakTimer(dt);

        let steering = this.playfightBehavior.getSteering(this.target);

        this.vehicle.steer(steering);
    }
}