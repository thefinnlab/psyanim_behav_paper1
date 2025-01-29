import PsyanimEntityPrefab from "../PsyanimEntityPrefab.js";

import PsyanimVehicle from "../components/steering/PsyanimVehicle.js";

import PsyanimMultiRaySensor from "../components/physics/PsyanimMultiRaySensor.js";
import PsyanimMultiRaySensorRenderer from "../components/rendering/PsyanimMultiRaySensorRenderer.js";
import PsyanimSeekBehavior from "../components/steering/PsyanimSeekBehavior.js";
import PsyanimObstacleAvoidanceBehavior from "../components/steering/PsyanimObstacleAvoidanceBehavior.js";

import PsyanimFleeBehavior from "../components/steering/PsyanimFleeBehavior.js";
import PsyanimEvadeBehavior from '../components/steering/PsyanimEvadeBehavior.js';
import PsyanimAdvancedFleeBehavior from "../components/steering/PsyanimAdvancedFleeBehavior.js";

import PsyanimWanderBehavior from "../components/steering/PsyanimWanderBehavior.js";
import PsyanimFOVSensor from "../components/physics/PsyanimFOVSensor.js";
import PsyanimBasicPreyBehavior from "../components/steering/PsyanimBasicPreyBehavior.js";
import PsyanimPreyAgent from '../components/steering/agents/PsyanimPreyAgent.js';

import PsyanimFOVRenderer from "../components/rendering/PsyanimFOVRenderer.js";

/**
 *  Prefab for creating a `Prey Agent`.
 * 
 *  A `Prey Agent` will wander until the target comes within its field-of-view and within the `panic distance`,
 *  at which point it will begin to flee from the target in a direction offset by the `subtlety` parameter.
 * 
 *  A `Prey Agent` has the following components: 
 * 
 *  `PsyanimVehicle`, `PsyanimFleeBehavior`, `PsyanimSeekBehavior`, `PsyanimWanderBehavior`, 
 *  `PsyanimFOVSensor`, `PsyanimBasicPreyBehavior`, `PsyanimPreyAgent`
 * 
 */
export default class PsyanimPreyPrefab extends PsyanimEntityPrefab {

    /** Prey params */

    /**
     *  Maximum number of degrees which this agent will offset it's flee direction.
     *  @type {Number}
     */
    subtlety;

    /**
     *  Determines how often the agent will recompute it's flee direction based on the `subtlety` parameter.
     *  @type {Number}
     */
    subtletyLag;

    /**
     *  Min. distance, in 'px', to target in which this predator will flee to maintain.
     *  @type {Number}
     */
    safetyDistance;

    /**
     *  If true, agent state transitions will be written out to Psyanim Debug Logs
     *  @type {boolean}
     */
    showDebugLogs;

    /**
     *  Toggles debug graphics for obstacle avoidance multi-ray sensor and field-of-view.
     *  @type {boolean}
     */
    showDebugGraphics;

    /** Flee params */

    /**
     *  Maximum speed this agent can flee from the target.
     *  @type {Number}
     */
    maxFleeSpeed;

    /**
     *  Maximum acceleration this agent can reach during flee from target.
     *  @type {Number}
     */
    maxFleeAcceleration;

    /**
     *  Distance from target at which the agent will begin fleeing.
     *  @type {Number}
     */
    panicDistance;

    /** Obstacle Avoidance Params */

    /**
     *  Max. speed the prey agent will seek away from screen boundaries
     *  @type {Number}
     */
    maxSeekSpeed;

    /**
     *  Max. acceleration the prey agent will reach when seeking away from screen boundaries
     *  @type {Number}
     */
    maxSeekAcceleration;

    /**
     *  Length of the main ray used for collision detection with obstacles.
     *  @type {Number}
     */
    mainRayLength;

    /**
     *  Length of the 'whisker' rays used for collision detection with obstacles.
     *  @type {Number}
     */
    whiskerLength;

    /**
     *  Angle of the 'whisker' rays with respect to the prey agent's forward direction
     *  @type {Number}
     */
    whiskerAngle;

    /** Wander params */

    /**
     *  Maximum speed at which the agent will wander.
     *  @type {Number}
     */
    maxWanderSpeed;

    /**
     *  Maximum acceleration the agent can attain during wander.
     *  @type {Number}
     */
    maxWanderAcceleration;

    /**
     *  Radius of the wander circle.
     *  @type {Number}
     */
    wanderRadius;

    /**
     *  Distance the wander circle is offset from the agent's position.
     *  @type {Number}
     */
    wanderOffset;

    /**
     *  Maximum number of degrees the wander target can move around the wander circle per frame.
     *  @type {Number}
     */
    maxWanderAngleChangePerFrame;

    /** Field-of-view Params */

    /**
     *  Determines how whether or not this agent will have a limited field of view.
     * 
     *  If false, the agent's field of view will be 360 degrees and have unlimited range.
     *  @type {boolean}
     */
    useFov;

    /**
     *  Field-of-view cone angle, in degrees.
     *  @type {Number}
     */
    fovAngle;

    /**
     *  Field-of-view cone height, in pixels.  Determines how far out the agent can see.
     *  @type {Number}
     */
    fovRange;

    /**
     *  Field-of-view resolution, in degrees.  
     * 
     *  This determines how many degrees of the field-of-view an entity must occupy to be detected.
     * 
     *  The lower the number, the higher the resolution, and the more CPU overhead this component has.
     * 
     *  @type {Number}
     */
    fovResolution;

    constructor(shapeParams = { isEmpty: true }, matterOptions = {}) {

        super(shapeParams, matterOptions);

        // prey params
        this.subtlety = 30;
        this.subtletyLag = 500;
        this.safetyDistance = 250;
        this.showDebugLogs = false;
        this.showDebugGraphics = false;

        // fov params
        this.useFov = false;
        this.fovAngle = 120;
        this.fovRange = 150;
        this.fovResolution = 5;

        // flee params
        this.maxFleeSpeed = 2;
        this.maxFleeAcceleration = 0.1;
        this.panicDistance = 250;

        // obstacle avoidance params
        this.maxSeekSpeed = 2;
        this.maxSeekAcceleration = 0.1;
        this.mainRayLength = 100;
        this.whiskerLength = 75;
        this.whiskerAngle = 25;

        // wander params
        this.maxWanderSpeed = 1.5;
        this.maxWanderAcceleration = 0.1;
        this.wanderRadius = 50;
        this.wanderOffset = 250;
        this.maxWanderAngleChangePerFrame = 10;
    }

    create(entity) {

        super.create(entity);

        let vehicle = entity.addComponent(PsyanimVehicle);
        vehicle.maxSpeed = 3;
        vehicle.turnSpeed = 0.05;

        let multiRaySensor = entity.addComponent(PsyanimMultiRaySensor);
        multiRaySensor.rayInfoList = [
            {
                id: 0,
                distance: this.mainRayLength,
                relativeAngle: 0
            },
            {
                id: 1,
                distance: this.whiskerLength,
                relativeAngle: this.whiskerAngle
            },
            {
                id: 2,
                distance: this.whiskerLength,
                relativeAngle: -this.whiskerAngle
            }
        ];

        if (this.showDebugGraphics)
        {
            let multiRaySensorRenderer = entity.addComponent(PsyanimMultiRaySensorRenderer);
            multiRaySensorRenderer.raySensor = multiRaySensor;    
        }

        let seek = entity.addComponent(PsyanimSeekBehavior);

        let obstacleAvoidanceBehavior = entity.addComponent(PsyanimObstacleAvoidanceBehavior);
        obstacleAvoidanceBehavior.multiRaySensor = multiRaySensor;
        obstacleAvoidanceBehavior.seekBehavior = seek;
        obstacleAvoidanceBehavior.maxSeekSpeed = this.maxSeekSpeed;
        obstacleAvoidanceBehavior.maxSeekAcceleration = this.maxSeekAcceleration;
        obstacleAvoidanceBehavior.avoidDistance = 2 * (this.whiskerLength * Math.sin(this.whiskerAngle * Math.PI / 180));

        let flee = entity.addComponent(PsyanimFleeBehavior);
        flee.maxSpeed = this.maxFleeSpeed;
        flee.maxAcceleration = this.maxFleeAcceleration;
        flee.panicDistance = this.panicDistance;

        let evade = entity.addComponent(PsyanimEvadeBehavior);
        evade.fleeBehavior = flee;
        evade.maxPredictionTime = 3.0;

        let advancedFlee = entity.addComponent(PsyanimAdvancedFleeBehavior);
        advancedFlee.fleeBehavior = evade;
        advancedFlee.obstacleAvoidanceBehavior = obstacleAvoidanceBehavior;

        let wander = entity.addComponent(PsyanimWanderBehavior);
        wander.seekBehavior = seek;
        wander.maxSeekSpeed = this.maxWanderSpeed;
        wander.maxSeekAcceleration = this.maxWanderAcceleration;
        wander.radius = this.wanderRadius;
        wander.offset = this.wanderOffset;
        wander.maxWanderAngleChangePerFrame = this.maxWanderAngleChangePerFrame;

        let fovSensor = null;

        if (this.useFov)
        {
            fovSensor = entity.addComponent(PsyanimFOVSensor);
            fovSensor.fovAngle = this.fovAngle;
            fovSensor.fovRange = this.fovRange;
            fovSensor.resolution = this.fovResolution;    
        }

        if (this.useFov && this.showDebugGraphics)
        {
            let fovRenderer = entity.addComponent(PsyanimFOVRenderer);
            fovRenderer.fovSensor = fovSensor;
        }

        let prey = entity.addComponent(PsyanimBasicPreyBehavior);
        prey.fleeBehavior = advancedFlee;
        prey.wanderBehavior = wander;
        prey.fovSensor = fovSensor;
        prey.subtlety = this.subtlety;
        prey.subtletyLag = this.subtletyLag;
        prey.safetyDistance = this.safetyDistance;
        prey.debug = this.showDebugLogs;

        let preyAgent = entity.addComponent(PsyanimPreyAgent);
        preyAgent.vehicle = vehicle;
        preyAgent.preyBehavior = prey;

        return entity;
    }
}