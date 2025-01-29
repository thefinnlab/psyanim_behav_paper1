import PsyanimEntityPrefab from '../PsyanimEntityPrefab.js';

import PsyanimVehicle from '../components/steering/PsyanimVehicle.js';
import PsyanimFleeBehavior from '../components/steering/PsyanimFleeBehavior.js';
import PsyanimEvadeBehavior from '../components/steering/PsyanimEvadeBehavior.js';
import PsyanimEvadeAgent from '../components/steering/agents/PsyanimEvadeAgent.js';

export default class PsyanimEvadeAgentPrefab extends PsyanimEntityPrefab {

    maxSpeed;
    maxAcceleration;

    constructor(shapeParams = { isEmpty: true }, matterOptions = {}) {

        super(shapeParams, matterOptions);

        this.maxSpeed = 6;
        this.maxAcceleration = 0.2;
    }

    create(entity) {

        super.create(entity);

        let evadeAgentVehicle = entity.addComponent(PsyanimVehicle);

        let fleeBehavior = entity.addComponent(PsyanimFleeBehavior);
        fleeBehavior.maxSpeed = this.maxSpeed;
        fleeBehavior.maxAcceleration = this.maxAcceleration;

        let evadeBehavior = entity.addComponent(PsyanimEvadeBehavior);
        evadeBehavior.fleeBehavior = fleeBehavior;

        let evadeAgent = entity.addComponent(PsyanimEvadeAgent);
        evadeAgent.vehicle = evadeAgentVehicle;
        evadeAgent.evadeBehavior = evadeBehavior;

        return entity;
    }
}