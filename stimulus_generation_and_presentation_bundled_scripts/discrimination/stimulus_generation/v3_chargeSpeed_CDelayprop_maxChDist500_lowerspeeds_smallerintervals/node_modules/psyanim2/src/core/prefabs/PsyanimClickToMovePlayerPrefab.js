import PsyanimEntityPrefab from "../PsyanimEntityPrefab.js";

import PsyanimPathfindingAgent from "../components/pathfinding/PsyanimPathfindingAgent.js";
import PsyanimPathfindingRenderer from "../components/rendering/PsyanimPathfindingRenderer.js";
import PsyanimVehicle from "../components/steering/PsyanimVehicle.js";

import PsyanimArriveBehavior from "../components/steering/PsyanimArriveBehavior.js";
import PsyanimArriveAgent from "../components/steering/agents/PsyanimArriveAgent.js";
import PsyanimClickToMove from "../components/controllers/PsyanimClickToMove.js";
import PsyanimPlayerController from "../components/controllers/PsyanimPlayerController.js";

export default class PsyanimClickToMovePlayerPrefab extends PsyanimEntityPrefab {

    grid;

    maxSpeed;
    innerDecelerationRadius;
    outerDecelerationRadius;

    debug;

    constructor(shapeParams = { isEmpty: true }, matterOptions = {}) {

        super(shapeParams, matterOptions);

        this.maxSpeed = 8;
        this.innerDecelerationRadius = 10;
        this.outerDecelerationRadius = 50;

        this.debug = false;
    }

    create(entity) {

        super.create(entity);

        // setup pathfinder
        let pathfinder = entity.addComponent(PsyanimPathfindingAgent);
        pathfinder.grid = this.grid;
        pathfinder.setDestination(entity.position);

        // setup path following
        let vehicle = entity.addComponent(PsyanimVehicle);

        let arriveBehavior = entity.addComponent(PsyanimArriveBehavior);
        arriveBehavior.maxSpeed = this.maxSpeed;
        arriveBehavior.innerDecelerationRadius = this.innerDecelerationRadius;
        arriveBehavior.outerDecelerationRadius = this.outerDecelerationRadius;

        let arriveAgent = entity.addComponent(PsyanimArriveAgent);
        arriveAgent.arriveBehavior = arriveBehavior;
        arriveAgent.vehicle = vehicle;

        let clickToMoveController = entity.addComponent(PsyanimClickToMove);
        clickToMoveController.grid = this.grid;
        clickToMoveController.pathfinder = pathfinder;
        clickToMoveController.arriveAgent = arriveAgent;

        let playerController = entity.addComponent(PsyanimPlayerController);
        playerController.clickToMoveController = clickToMoveController;

        if (this.debug)
        {
            // setup path renderer
            let pathfindingRenderer = entity.addComponent(PsyanimPathfindingRenderer);
            pathfindingRenderer.pathfinder = pathfinder;
            pathfindingRenderer.setGridVisible(true);
            pathfindingRenderer.radius = 4;
        }

        return entity;
    }
}